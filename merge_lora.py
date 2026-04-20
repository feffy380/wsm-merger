# validation code adapted from https://github.com/spacepxl/demystifying-sd-finetuning

import math
import random
from argparse import ArgumentParser
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import diffusers
import matplotlib.pyplot as plt
import numpy as np
import safetensors
import torch
from diffusers.training_utils import compute_snr
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

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


class LoraManager:
    def __init__(self, lora_paths, device="cuda"):
        self.lora_infos = [LoraInfo(path) for path in lora_paths]
        self.lora_infos.sort(key=lambda item: item.steps())
        self.loras = [safetensors.safe_open(lora_info.path, framework="pt", device=device) for lora_info in self.lora_infos]

    def __len__(self):
        return len(self.loras)

    def __getitem__(self, idx):
        return self.loras[idx], self.lora_infos[idx]

    def merge_range(self, start, end, decay_type):
        # derive merge weights according to Theorem 3.1 from the paper
        window = end - start + 1
        k = window - 1
        t = np.linspace(0, 1, window+1)
        if decay_type == "1-sqrt":
            w = 1 - np.sqrt(t)
        elif decay_type == "linear":
            w = 1 - t
        else:
            raise ValueError(f"Unknown decay type: {decay_type}")
        weights = np.concat([[1-w[1]], w[1:k] - w[2:k+1], [w[k]]])

        state_dict = {}
        for lora, weight in zip(self.loras[start:end+1], weights):
            for key in lora.keys():
                if "alpha" in key:
                    if key not in state_dict:
                        state_dict[key] = lora.get_tensor(key)
                elif key not in state_dict:
                    state_dict[key] = lora.get_tensor(key) * weight
                else:
                    state_dict[key].add_(lora.get_tensor(key) * weight)

        return state_dict, self.lora_infos[start:end+1]


class CenterOutStrategy:
    """
    Finds the single best point, then grows outward greedily.
    Works best with smooth validation curve.
    Skip first phase by providing an initial point.
    """
    def __init__(self, n, anchor=None):
        self.n = n
        self.anchor = anchor
        self.next_anchor = 0 if anchor is None else n
        self.start = 0
        self.end = 0
        self.step_size = 1
        self.grow_direction = 0
        self.best_loss = float("inf")
        self.left_finished = False
        self.right_finished = False

    def is_finished(self):
        return self.left_finished and self.right_finished

    def get_candidates(self):
        """
        Expand window left and right. None if boundary hit.

        Returns list of (start, end) merge windows
        """
        if self.next_anchor < self.n:
            # phase 1: find starting point with lowest loss
            return [(self.next_anchor, self.next_anchor)]
        else:
            # phase 2: grow outward
            left = (self.start - self.step_size, self.end) if self.start - self.step_size >= 0 else None
            right = (self.start, self.end + self.step_size) if self.end + self.step_size < self.n else None

            if self.grow_direction > 0:
                window = right
                if window is None:
                    window = (self.start, self.n - 1)
                if window[1] == self.n - 1:
                    self.right_finished = True
            elif self.grow_direction < 0:
                window = left
                if window is None:
                    window = (0, self.end)
                if window[0] == 0:
                    self.left_finished = True

            return [window]

    def update(self, losses: list):
        """
        Moves boundaries to the best performing side

        Returns current best ((start, end), loss)
        """
        # TODO: grow one direction at a time because merge window tends to be right of the lowest point
        if self.grow_direction == 0:
            # phase 1: find starting point with lowest loss
            loss = losses[0]
            if loss < self.best_loss:
                self.anchor = self.next_anchor
                self.best_loss = loss
            self.next_anchor += 1
            if self.next_anchor >= self.n:
                # finished. set window to anchor and start second phase
                self.start = self.anchor
                self.end = self.anchor
                self.grow_direction = 1
            return (self.anchor, self.anchor), self.best_loss
        else:
            # phase 2: grow outward
            loss = losses[0]
            if self.grow_direction < 0 and loss < self.best_loss:
                # grow left
                self.start -= self.step_size
                self.best_loss = loss
                self.step_size = 1
            elif loss < self.best_loss:
                # grow right
                self.end += self.step_size
                self.best_loss = loss
                self.step_size = 1
            else:
                # neither better.
                # flip direction or increase step size
                if self.grow_direction > 0:
                    if self.left_finished:
                        self.step_size *= 2
                    else:
                        self.grow_direction *= -1
                else:
                    self.step_size *= 2
                    if not self.right_finished:
                        self.grow_direction *= -1
            return (self.start, self.end), self.best_loss


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


def validate(args, pipeline, val_dataloader):
    unet: diffusers.UNet2DConditionModel = pipeline.unet
    noise_scheduler = pipeline.scheduler

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

    NUM_VAL_TIMESTEPS = args.val_num_timesteps  # 4 -> 200, 400, 600, 800
    val_timesteps = np.linspace(0, noise_scheduler.config.num_train_timesteps, (NUM_VAL_TIMESTEPS + 2), dtype=int)[1:-1]
    val_total_steps = NUM_VAL_TIMESTEPS * len(val_dataloader)

    val_loss = 0.0
    with torch.inference_mode():
        for batch in val_dataloader:
            batch["latent"] = batch["latent"].to(args.device)
            batch["prompt_embeds"] = batch["prompt_embeds"].to(args.device)
            batch["pooled_prompt_embeds"] = batch["pooled_prompt_embeds"].to(args.device)
            for timestep in val_timesteps:
                loss = get_pred(batch, timesteps=[timestep])
                val_loss += loss.detach().item()

    return val_loss / val_total_steps


def save_chart(points, optimum, outdir):
    x, y = zip(*points)
    plt.plot(x, y, label="val_loss")

    # star at best point
    best_window, best_val = optimum
    star_xy = (best_window[1], best_val)
    plt.plot(*star_xy, marker="*", color="gold", markersize=15, label=f"{best_window[0]}-{best_window[1]}")
    plt.annotate(str(round(best_val, 5)), star_xy)

    # formatting
    plt.title(outdir.stem)
    plt.xlabel("Iterations")
    plt.ylabel("Validation loss")
    plt.legend()
    plt.grid(True)

    # save as png
    plt.savefig(outdir / "merge-results.png", dpi=300, bbox_inches="tight")


def main(args):
    dataset_path = Path(args.dataset_path).expanduser()
    ckpt_path = Path(args.ckpt_path).expanduser()

    if args.range is None:
        pipeline = diffusers.StableDiffusionXLPipeline.from_single_file(ckpt_path, torch_dtype=torch.float16, local_files_only=True).to(args.device)
        pipeline.vae.config.force_upcast = False
        pipeline.unet.requires_grad_(False)
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.text_encoder_2.requires_grad_(False)
        pipeline.scheduler = diffusers.DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            prediction_type=args.prediction_type,
            rescale_betas_zero_snr=(args.prediction_type == "v_prediction"),
        )

        val_dataset = LatentDataset(dataset_path, pipeline, resolution=args.resolution)
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

    lora_dir = Path(args.lora_dir).expanduser()
    lora_paths = list(lora_dir.glob("*-step*.safetensors"))
    lora_manager = LoraManager(lora_paths, device=args.device)
    best_val = float("inf")
    best_components = None
    best_lora = None

    outpath = lora_dir / f"{lora_dir.stem}-merged.safetensors"
    metadata = {
        "val_seed": str(args.val_seed),
        "decay_type": args.decay_type,
        "prediction_type": args.prediction_type,
        "min_snr_gamma": str(args.min_snr_gamma),
    }

    if args.range is not None:
        # manual merge
        merge_window = [i for i, info in enumerate(lora_manager.lora_infos) if info.steps() in args.range]
        if len(merge_window) != 2:
            raise ValueError(f"--range {args.range} is not a valid range of checkpoints")
        best_lora, best_components = lora_manager.merge_range(*merge_window, decay_type=args.decay_type)
        best_val = None
        print(f"manual merge: steps {best_components[0].steps()}-{best_components[-1].steps()}")
    else:
        # automatic merge
        search_strategy = CenterOutStrategy(len(lora_manager))
        results = []
        while not search_strategy.is_finished():
            merge_windows = search_strategy.get_candidates()
            losses = []
            for merge_window in merge_windows:
                if merge_window is None:
                    losses.append(None)
                    continue

                lora_sd, components = lora_manager.merge_range(*merge_window, decay_type=args.decay_type)

                # Calculate validation loss for the merge
                pipeline.unload_lora_weights()
                pipeline.load_lora_weights(lora_sd)
                with temp_rng(args.val_seed):
                    val_loss = validate(args, pipeline, val_dataloader)
                losses.append(val_loss)

                if val_loss < best_val:
                    best_val = val_loss
                    best_lora = lora_sd
                    best_components = components
                    # save best so far in case of interruption
                    metadata["val_loss"] = str(best_val)
                    metadata["components"] = ",".join([str(c.steps()) for c in best_components])
                    safetensors.torch.save_file(best_lora, outpath, metadata)

                print(f"steps {components[0].steps()}-{components[-1].steps()}: {val_loss}")

                # save results for validation curve
                if merge_window[0] == merge_window[1]:
                    results.append((components[0].steps(), val_loss))
            search_strategy.update(losses)

        print(f"best: steps {best_components[0].steps()}-{best_components[-1].steps()}: {best_val}")

        best_window = (best_components[0].steps(), best_components[-1].steps())
        save_chart(results, (best_window, best_val), lora_dir)

    metadata["val_loss"] = str(best_val)
    metadata["components"] = ",".join([str(c.steps()) for c in best_components])
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
        "--lora-dir", type=str, required=True,
        help="Path to dir with numbered LoRA checkpoints e.g., lora-step1234.safetensors",
    )
    parser.add_argument(
        "--resolution", type=int, default=1024,
        help="Resolution to use for validation",
    )
    parser.add_argument(
        "--decay-type", "-d", default="1-sqrt",
        choices=["1-sqrt", "linear"],
        help="LR decay schedule to use for merging: [1-sqrt, linear] (default: 1-sqrt)",
    )
    parser.add_argument(
        "--range", "-r", type=int, nargs=2,
        help="Merge a specified range of checkpoints",
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
    parser.add_argument(
        "--val-num-timesteps", "-t", type=int, default=4,
        help="Number of timesteps to use to calculate validation loss (default: 4)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")

    args = parser.parse_args()
    main(args)
