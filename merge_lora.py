# Parts adapted from https://github.com/spacepxl/demystifying-sd-finetuning

from dataclasses import dataclass
import math
import random
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path

import diffusers
import numpy as np
import torch
from diffusers.training_utils import compute_snr
from PIL import Image
import safetensors.torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm.auto import tqdm

diffusers.logging.set_verbosity_error()


@dataclass
class LoraInfo:
    path: Path

    def name(self) -> str:
        return self.path.stem

    def steps(self) -> int:
        return int(self.name().split("step")[-1])

    def __repr__(self):
        return self.name()


class LatentDataset(Dataset):
    def __init__(self, root_folder, pipeline, resolution=1024):
        self.root_folder = root_folder
        self.resolution = resolution
        self.batches = []

        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(pipeline.vae.dtype, scale=True),
            v2.Normalize([0.5], [0.5]),
        ])

        print("Caching validation data")
        exts = {".png", ".jpg", ".jpeg"}
        for image_path in Path(root_folder).iterdir():
            if image_path.suffix.lower() not in exts:
                continue

            latent, crop_top_left, original_size = self.preprocess_image(image_path, pipeline.vae)
            caption = image_path.with_suffix(".txt").read_text().strip()
            prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(caption, device=pipeline.text_encoder.device, do_classifier_free_guidance=False)
            batch = {
                "latent": latent.squeeze().cpu(),
                "crop_top_left": crop_top_left,
                "original_size": original_size,
                "prompt_embeds": prompt_embeds.squeeze().cpu(),
                "pooled_prompt_embeds": pooled_prompt_embeds.squeeze().cpu(),
            }
            self.batches.append(batch)

    def get_bucket(self, image: Image):
        """Calculate bucket dimensions with closest aspect ratio to original"""
        w, h = image.size
        ar = w / h
        base = 64

        target_area = self.resolution ** 2
        W = (target_area * ar)**0.5
        H = (target_area / ar)**0.5

        ar_errors = []
        WHs = []
        for f in [math.floor, math.ceil]:
            for g in [math.floor, math.ceil]:
                W2 = int(base * f(W / base))
                H2 = int(base * g(H / base))
                ar_errors.append(W2 / H2)
                WHs.append((W2, H2))

        ar_errors = np.log(np.array(ar_errors)) - np.log(ar)
        idx = np.abs(ar_errors).argmin()
        return WHs[idx]

    def crop_to_bucket(self, image: Image.Image):
        w, h = image.size
        bucket = self.get_bucket(image)

        # resize short side to fit bucket
        ar = w / h
        ar_bucket = bucket[0] / bucket[1]
        if ar > ar_bucket:
            scale = bucket[1] / h
        else:
            scale = bucket[0] / w
        resized_size = (int(w * scale + 0.5), int(h * scale + 0.5))
        image = image.resize(resized_size)

        # trim excess
        w, h = image.size
        assert w == bucket[0] or h == bucket[1]
        ltrb = (0, 0, 0, 0)
        if w > bucket[0]:
            trim_size = w - bucket[0]
            p = trim_size // 2
            ltrb = (p, 0, p + bucket[0], bucket[1])
            image = image.crop(ltrb)
        if h > bucket[1]:
            trim_size = h - bucket[1]
            p = trim_size // 2
            ltrb = (0, p, bucket[0], p + bucket[1])
            image = image.crop(ltrb)

        return image, ltrb

    def preprocess_image(self, image_path, vae: diffusers.AutoencoderKL):
        img = Image.open(image_path).convert("RGB")
        img, ltrb = self.crop_to_bucket(img)
        pixels = self.transforms(img)  # hw
        latent = vae.encode(pixels.to(vae.device, dtype=vae.dtype).unsqueeze(0)).latent_dist.sample()
        latent = latent * vae.config.scaling_factor
        return latent, (ltrb[1], ltrb[0]), img.size[::-1]

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


@contextmanager
def temp_rng(new_seed=None):
    """
    https://github.com/fpgaminer/bigasp-training/blob/main/utils.py#L73
    Context manager that saves and restores the RNG state of PyTorch, NumPy and Python.
    If new_seed is not None, the RNG state is set to this value before the context is entered.
    """

    # Save RNG state
    old_torch_rng_state = torch.get_rng_state()
    old_torch_cuda_rng_state = torch.cuda.get_rng_state()
    old_numpy_rng_state = np.random.get_state()
    old_python_rng_state = random.getstate()

    # Set new seed
    if new_seed is not None:
        torch.manual_seed(new_seed)
        torch.cuda.manual_seed(new_seed)
        np.random.seed(new_seed)
        random.seed(new_seed)

    yield

    # Restore RNG state
    torch.set_rng_state(old_torch_rng_state)
    torch.cuda.set_rng_state(old_torch_cuda_rng_state)
    np.random.set_state(old_numpy_rng_state)
    random.setstate(old_python_rng_state)


def apply_snr_weight(loss: torch.Tensor, timesteps: torch.IntTensor, noise_scheduler, gamma, prediction_type="epsilon"):
    # soft-min-snr
    snr = compute_snr(noise_scheduler, timesteps).float().to(loss.device)
    snr_weight = snr * gamma / (snr + gamma)
    if prediction_type == "v_prediction":
        snr_weight = snr_weight / (snr + 1)
    else:
        snr_weight = snr_weight / snr
    loss = loss * snr_weight
    return loss


def validation(args, pipeline, val_dataloader):
    unet: diffusers.UNet2DConditionModel = pipeline.unet
    vae: diffusers.AutoencoderKL = pipeline.vae
    noise_scheduler = diffusers.DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        clip_sample=False,
        prediction_type=args.prediction_type,
        rescale_betas_zero_snr=(args.prediction_type == "v_prediction"),
    )

    def get_pred(batch, timesteps):
        latents = batch["latent"]
        prompt_embeds = batch["prompt_embeds"]
        pooled_prompt_embeds = batch["pooled_prompt_embeds"]

        # encode inputs
        timesteps = torch.tensor(timesteps, dtype=torch.long, device=args.device)

        # get model prediction and target
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        model_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds,
                "time_ids": pipeline._get_add_time_ids(
                    original_size=batch["original_size"],
                    crops_coords_top_left=batch["crop_top_left"],
                    target_size=batch["original_size"],
                    dtype=prompt_embeds.dtype,
                    text_encoder_projection_dim=pipeline.text_encoder_2.config.projection_dim,
                ).to(args.device),
            },
            return_dict=False,
        )[0]
        if noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

        # calculate loss
        loss = torch.nn.functional.mse_loss(model_pred, target)
        if args.min_snr_gamma is not None:
            loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, args.prediction_type)
        return loss

    NUM_VAL_TIMESTEPS = 4  # 200, 400, 600, 800
    val_timesteps = np.linspace(0, noise_scheduler.config.num_train_timesteps, (NUM_VAL_TIMESTEPS + 2), dtype=int)[1:-1]
    val_total_steps = NUM_VAL_TIMESTEPS * len(val_dataloader)

    val_loss = 0.0
    with torch.inference_mode(), temp_rng(args.val_seed):
        for batch in val_dataloader:
            batch["latent"] = batch["latent"].to(args.device)
            batch["prompt_embeds"] = batch["prompt_embeds"].to(args.device)
            batch["pooled_prompt_embeds"] = batch["pooled_prompt_embeds"].to(args.device)
            for timestep in val_timesteps:
                loss = get_pred(batch, timesteps=[timestep])
                val_loss += loss.detach().item()

    return val_loss / val_total_steps


def create_merges(lora_paths, merge_window, window_stride=1):
    loras = [safetensors.torch.load_file(lora_path.path, device="cuda") for lora_path in lora_paths]
    # TODO: different merge strategies like 1-sqrt(t)
    weights = [1 / merge_window] * merge_window
    window_start = 0
    while window_start + merge_window  <= len(loras):
        state_dict = {}
        subset = loras[window_start:window_start+merge_window]
        for lora, weight in zip(subset, weights):
            for key in lora.keys():
                if "alpha" in key:
                    if key not in state_dict:
                        state_dict[key] = lora[key]
                elif key not in state_dict:
                    state_dict[key] = lora[key] * weight
                else:
                    state_dict[key].add_(lora[key] * weight)

        yield state_dict, lora_paths[window_start:window_start+merge_window]

        if window_start + merge_window == len(loras):
            break
        window_start += window_stride
        # clamp if stride would make us skip a window
        if window_start + merge_window > len(loras):
            window_start = len(loras) - merge_window


def main(args):
    dataset_path = Path(args.dataset_path).expanduser()
    ckpt_path = Path(args.ckpt_path).expanduser()
    # TODO: generate chart

    pipeline = diffusers.StableDiffusionXLPipeline.from_single_file(ckpt_path, torch_dtype=torch.float16, local_files_only=True).to(args.device)
    pipeline.vae.config.force_upcast = False
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)

    val_dataset = LatentDataset(dataset_path, pipeline)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    lora_dir = Path(args.lora_path).expanduser()
    lora_paths = [LoraInfo(path) for path in lora_dir.glob("*-step*.safetensors")]
    lora_paths.sort(key=lambda item: item.steps())
    best_val = float("inf")
    best_components = None
    best_lora = None
    # TODO: exhaustive search
    for lora_sd, components in create_merges(lora_paths, args.merge_window, args.window_stride):
        pipeline.unload_lora_weights()
        pipeline.load_lora_weights(lora_sd)
        val_loss = validation(args, pipeline, val_dataloader)
        print(f"steps {components[0].steps()}-{components[-1].steps()}: {val_loss}")
        if val_loss < best_val:
            best_val = val_loss
            best_components = components
            best_lora = lora_sd
    print(f"best: steps {best_components[0].steps()}-{best_components[-1].steps()}: {best_val}")

    outpath = lora_dir / f"{lora_dir.stem}-merged.safetensors"
    metadata = {
        "components": ",".join([str(c.steps()) for c in best_components]),
        "val_loss": str(val_loss),
        "val_seed": str(args.val_seed),
        "prediction_type": args.prediction_type,
        "min_snr_gamma": str(args.min_snr_gamma),
        "merge_window": str(args.merge_window),
        "window_stride": str(args.window_stride),
    }
    safetensors.torch.save_file(best_lora, outpath, metadata)
    print(f"Wrote merged LoRA to {outpath}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--dataset-path", type=str, required=True,
        help="Path to validation dataset (.png or .jpg with matching .txt captions)",
    )
    parser.add_argument(
        "--ckpt-path", type=str, required=True,
        help="Path to single file SDXL checkpoint",
    )
    parser.add_argument(
        "--lora-path", type=str, required=True,
        help="Path to dir with numbered LoRA checkpoints e.g., lora-step1234.safetensors",
    )
    parser.add_argument(
        "--merge-window", "-w", type=int, default=4,
        help="Merge window size (number of checkpoints to merge)",
    )
    parser.add_argument(
        "--window-stride", "-s", type=int, default=1,
        help="Merge window stride",
    )
    parser.add_argument(
        "--prediction-type", "-p", type=str, default="epsilon",
        help="Prediction type: [epsilon, v_prediction]",
    )
    parser.add_argument(
        "--min-snr-gamma", type=float, required=False,
        help="Apply Min-SNR-Gamma loss weighting",
    )
    parser.add_argument(
        "--val-seed", type=int, default=380,
        help="Validation random seed",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")

    args = parser.parse_args()
    main(args)
