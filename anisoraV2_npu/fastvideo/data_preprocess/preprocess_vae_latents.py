from fastvideo.dataset import getdataset
from torch.utils.data import DataLoader
from fastvideo.utils.dataset_utils import Collate
import argparse

import os
import torch
if os.getenv("ACCELERATOR") == "npu":
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.config.allow_internal_format = False
    print("import torch_npu\n")
else:
    print("Warning: Missing torch_npu\n")

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import json
import os
from diffusers import AutoencoderKLMochi
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from fastvideo.utils.load import load_vae
from tqdm import tqdm

from fastvideo.bili_space.boss import BossClient

logger = get_logger(__name__)


def main(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size, "local rank", local_rank)

    encoder_device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=local_rank
        )

    if os.environ.get("VS_DEBUG", default="false").lower() == "true" and torch.distributed.get_rank() == 0:
        import debugpy
        debugpy.listen(5678)
        print("[RANK 0] Waiting for debugger attach")
        debugpy.wait_for_client()

    boss_handle = BossClient.get_client()
    train_dataset = getdataset(args)
    sampler = DistributedSampler(
        train_dataset, rank=local_rank, num_replicas=world_size, shuffle=False, drop_last=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    vae, autocast_type, fps = load_vae(args.model_type, args.model_path)
    vae.enable_tiling()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "latent"), exist_ok=True)

    json_data = []
    for _, data in tqdm(enumerate(train_dataloader), disable=local_rank != 0):
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=autocast_type):
                latents = vae.encode(data["pixel_values"].to(encoder_device))[
                    "latent_dist"
                ].sample()
            for idx, video_path in enumerate(data["path"]):
                video_name = os.path.basename(video_path).split(".")[0]
                latent_path = os.path.join(
                    args.output_dir, "latent", video_name + ".pt"
                )
                torch.save(latents[idx].to(torch.bfloat16), latent_path)

                if boss_handle:
                    relative_path = os.path.relpath(latent_path, args.output_dir)
                    boss_handle.upload_file(os.path.abspath(latent_path), relative_path)

                item = {}
                item["length"] = latents[idx].shape[1]
                item["latent_path"] = video_name + ".pt"
                item["caption"] = data["text"][idx]
                json_data.append(item)
                print(f"{video_name} processed")
    dist.barrier()
    local_data = json_data
    gathered_data = [None] * world_size
    dist.all_gather_object(gathered_data, local_data)
    if local_rank == 0:
        all_json_data = [item for sublist in gathered_data for item in sublist]
        with open(os.path.join(args.output_dir, "videos2caption_temp.json"), "w") as f:
            json.dump(all_json_data, f, indent=4)
        if boss_handle:
            src_file = os.path.abspath(os.path.join(args.output_dir, "videos2caption_temp.json"))
            boss_handle.upload_file(src_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--model_path", type=str, default="data/mochi")
    parser.add_argument("--model_type", type=str, default="mochi")
    parser.add_argument("--data_merge_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_latent_t", type=int, default=28, help="Number of latent timesteps."
    )
    parser.add_argument("--max_height", type=int, default=480)
    parser.add_argument("--max_width", type=int, default=848)
    parser.add_argument("--video_length_tolerance_range", type=int, default=2.0)
    parser.add_argument("--group_frame", action="store_true")  # TODO
    parser.add_argument("--group_resolution", action="store_true")  # TODO
    parser.add_argument("--dataset", default="t2v")
    parser.add_argument("--train_fps", type=int, default=30)
    parser.add_argument("--use_image_num", type=int, default=0)
    parser.add_argument("--text_max_length", type=int, default=256)
    parser.add_argument("--speed_factor", type=float, default=1.0)
    parser.add_argument("--drop_short_ratio", type=float, default=1.0)
    # text encoder & vae & diffusion model
    parser.add_argument("--text_encoder_name", type=str, default="google/t5-v1_1-xxl")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    args = parser.parse_args()
    main(args)
