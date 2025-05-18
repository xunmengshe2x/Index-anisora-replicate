import argparse
from email.policy import strict
import logging
import math
import os
import shutil
from pathlib import Path

import torch
import torch.distributed
if os.getenv("ACCELERATOR", default="") == "npu":
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.config.allow_internal_format = False
    print("import torch_npu\n")
else:
    print("Warning: Missing torch_npu\n")

from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
)
from fastvideo.utils.communications import sp_parallel_dataloader_wrapper, broadcast
from fastvideo.models.mochi_hf.mochi_latents_utils import normalize_dit_input
from fastvideo.utils.validation import log_validation
import time
from torch.utils.data import DataLoader
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
import json
from torch.utils.data.distributed import DistributedSampler
from fastvideo.utils.dataset_utils import LengthGroupedSampler
import wandb
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers import FlowMatchEulerDiscreteScheduler
from fastvideo.utils.load import load_transformer
from diffusers.optimization import get_scheduler
from fastvideo.models.mochi_hf.modeling_mochi import MochiTransformer3DModel
from diffusers.utils import check_min_version
from fastvideo.dataset.latent_datasets import LatentDataset, latent_collate_function
import torch.distributed as dist
from safetensors.torch import save_file, load_file
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from fastvideo.utils.checkpoint import (
    save_checkpoint,
    save_lora_checkpoint,
    resume_lora_optimizer,
)
from fastvideo.utils.logging_ import main_print
from fastvideo.models.mochi_hf.pipeline_mochi import MochiPipeline

from fastvideo.bili_space.wan.fm_scheduler import FlowMatchScheduler

from fastvideo.bili_space.log_print import print_parser_val, report_memory
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")
import time
from collections import deque
from functools import wraps

experimental_config = torch_npu.profiler._ExperimentalConfig(export_type=torch_npu.profiler.ExportType.Text, \
                                                             data_simplification=False, \
                                                             aic_metrics=torch_npu.profiler.AiCMetrics.ArithmeticUtilization, \
                                                             profiler_level=torch_npu.profiler.ProfilerLevel.Level1)

prof_schedule = torch_npu.profiler.schedule(# During this phase profiler is not active.
                                            wait=2, \
                                            # During this phase profiler starts tracing, but the results are discarded.
                                            warmup=1, \
                                            # During this phase profiler traces and records data.
                                            active=1, \
                                            # Specifies an upper bound on the number of cycles.
                                            repeat=1)
torch_profiler = torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU, \
                                                        torch_npu.profiler.ProfilerActivity.NPU], \
                                            profile_memory=True, with_stack=True, \
                                            schedule=prof_schedule, \
                                            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./npu_prof_result"), \
                                            experimental_config=experimental_config)


def profiler_tools(prof=None):
    def profiler_wrap(f):
        if prof != None:
            main_print(f"Enable profiler_tools.")
            prof.start()
        else:
            main_print(f"Disable profiler_tools.")
        @wraps(f)
        def decorated(*args, **kwarg):
            loss, grad_norm = f(*args, **kwarg)
            if prof != None:
                prof.step()
            return loss, grad_norm
        return decorated
    return profiler_wrap


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    generator,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(
            mean=logit_mean,
            std=logit_std,
            size=(batch_size,),
            device="cpu",
            generator=generator,
        )
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu", generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu", generator=generator)
    return u


def get_sigmas(noise_scheduler, device, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


# prof=torch_profiler
@profiler_tools(prof=None)
def train_one_step(
    transformer,
    model_type,
    optimizer,
    lr_scheduler,
    loader,
    noise_scheduler,
    noise_random_generator,
    gradient_accumulation_steps,
    sp_size,
    precondition_outputs,
    max_grad_norm,
    weighting_scheme,
    logit_mean,
    logit_std,
    mode_scale,
    extra=None,now_step=0,args=None
):
    total_loss = 0.0
    optimizer.zero_grad()
    for _ in range(gradient_accumulation_steps):
        (
            latents,
            encoder_hidden_states,
            latents_attention_mask,
            encoder_attention_mask,
        ) = latent_pkg = next(loader)
        latents, tensor_dict = normalize_dit_input(model_type, latents, latent_pkg)
        batch_size = latents.shape[0]

        if model_type == "wan":
            assert tensor_dict != None
            device = latents.device
            vae1todo = tensor_dict["vae1todo"]
            t5 = tensor_dict["t5"]
            clip = tensor_dict["clip"]
            noise = tensor_dict["noise"]

            set_seedd(args.seed, now_step)
            timestep_id = torch.randint(0, noise_scheduler.num_train_timesteps, (1,))
            timestep = noise_scheduler.timesteps[timestep_id].to(device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timestep)
            training_target = noise_scheduler.training_target(latents, noise, timestep)
            # Sync data with sp.
            if nccl_info.sp_size > 1:
                broadcast(t5)
                broadcast(clip)
                broadcast(vae1todo)
                broadcast(noisy_latents)
                broadcast(training_target)
                broadcast(timestep)


            with torch.autocast("cuda", torch.bfloat16):
                arg_c = {'context':[t5[0]], 'clip_fea':clip, 'seq_len':math.prod(noisy_latents.shape[-3:]) // 4, \
                         'y':[vae1todo], "is_train":True, "nccl_info":nccl_info}
                # print(111111, vae1todo.shape, nccl_info.sp_size,arg_c["seq_len"])
                '''
                        46800=13*80*45
                        80*16=1280
                        45*16=720
                        480/16=30
                        832/16=52
                '''
                noise_pred = transformer([noisy_latents[0]], t=timestep,**arg_c)[0]

            training_target = training_target.squeeze(0)
            set_seedd(args.seed, now_step)
            loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
            # I think gradient_accumulation_steps is error
            loss = loss * noise_scheduler.training_weight(timestep) / gradient_accumulation_steps
            set_seedd(args.seed, now_step)
        elif model_type == "cog15B" and extra != None:
            assert "loss_fn" in extra
            assert tensor_dict != None
            loss_fn = extra["loss_fn"]
            scale_factor = extra.get("scale_factor", 1.0)
            # video/image latent.
            latents *= scale_factor
            # Text prompt.
            encoder_hidden_states = tensor_dict.get("crossattn", None)
            vector_in = tensor_dict.get("vector", None)
            assert encoder_hidden_states != None
            assert vector_in != None
            # Sync data with sp.
            if nccl_info.sp_size > 1:
                broadcast(latents)
                broadcast(encoder_hidden_states)
                broadcast(vector_in)

            batch_in = {"crossattn": encoder_hidden_states, "vector": vector_in, "sp_cfg": nccl_info}
            with torch.autocast("cuda", torch.bfloat16):
                loss = loss_fn(network=transformer, input=latents, batch=batch_in)
        else:
            noise = torch.randn_like(latents)
            u = compute_density_for_timestep_sampling(
                weighting_scheme=weighting_scheme,
                batch_size=batch_size,
                generator=noise_random_generator,
                logit_mean=logit_mean,
                logit_std=logit_std,
                mode_scale=mode_scale,
            )
            indices = (u * noise_scheduler.config.num_train_timesteps).long()
            timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)
            if sp_size > 1:
                # Make sure that the timesteps are the same across all sp processes.
                broadcast(timesteps)
            sigmas = get_sigmas(
                noise_scheduler,
                latents.device,
                timesteps,
                n_dim=latents.ndim,
                dtype=latents.dtype,
            )
            noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
            # if rank<=0:
            #     print("2222222222222222222222222222222222222222222222")
            # print(type(latents_attention_mask))
            # print(latents_attention_mask)
            with torch.autocast("cuda", torch.bfloat16):
                model_pred = transformer(
                    noisy_model_input,
                    encoder_hidden_states,
                    timesteps,
                    encoder_attention_mask,  # B, L
                    return_dict=False,
                )[0]
            # if rank<=0:
            #     print("333333333333333333333333333333333333333333333333")
            if precondition_outputs:
                model_pred = noisy_model_input - model_pred * sigmas
            if precondition_outputs:
                target = latents
            else:
                target = noise - latents

            loss = (
                torch.mean((model_pred.float() - target.float()) ** 2)
                / gradient_accumulation_steps
            )

        loss.backward()
        avg_loss = loss.detach().clone()
        # dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(avg_loss)
        with torch.no_grad():
            avg_loss = avg_loss / torch.distributed.get_world_size()
        total_loss += avg_loss.item()

    grad_norm = transformer.clip_grad_norm_(max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    return total_loss, grad_norm.item()


def set_seedd(seed,now_step):
    now_seed=seed + int(os.environ["RANK"]) // nccl_info.sp_size+now_step*368
    if os.getenv("ACCELERATOR", default="") == "npu":
        torch_npu.npu.manual_seed_all(now_seed)
        torch_npu.npu.manual_seed(now_seed)
        torch.use_deterministic_algorithms(True)
    set_seed(now_seed,deterministic=True)

def main(parser):
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    if os.environ.get("VS_DEBUG", default="false").lower() == "true" and torch.distributed.get_rank() == 0:
        import debugpy
        debugpy.listen(5678)
        print("[RANK 0] Waiting for debugger attach")
        debugpy.wait_for_client()



    report_memory("Start:")
    print_parser_val(parser)
    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seedd(args.seed,0)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.
    noise_random_generator = None

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.

    # Create model:
    main_print(f"--> loading model from {args.pretrained_model_name_or_path if args.pretrained_model_name_or_path else args.model_type}")
    # keep the master weight to float32
    transformer, extra_rtv = load_transformer(
        args.model_type,
        args.dit_model_name_or_path,
        args.pretrained_model_name_or_path,
        torch.float32 if args.master_weight_type == "fp32" else torch.bfloat16,
        args.dit_model_cfg
    )

    if args.use_lora:
        assert args.model_type == "mochi", "LoRA is only supported for Mochi model."
        transformer.requires_grad_(False)
        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights=True,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        transformer.add_adapter(transformer_lora_config)

    if args.resume_from_lora_checkpoint:
        lora_state_dict = MochiPipeline.lora_state_dict(
            args.resume_from_lora_checkpoint
        )
        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(
            transformer, transformer_state_dict, adapter_name="default"
        )
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                main_print(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

    main_print(
        f"  Total training parameters = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e6} M"
    )
    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        args.use_lora,
        args.use_cpu_offload,
        args.master_weight_type,
    )

    if args.use_lora:
        transformer.config.lora_rank = args.lora_rank
        transformer.config.lora_alpha = args.lora_alpha
        transformer.config.lora_target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        transformer._no_split_modules = [
            no_split_module.__name__ for no_split_module in no_split_modules
        ]
        fsdp_kwargs["auto_wrap_policy"] = fsdp_kwargs["auto_wrap_policy"](transformer)
    transformer = FSDP(transformer, **fsdp_kwargs,)
    main_print(f"--> model loaded")
    report_memory("transformer loaded:")

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer, no_split_modules, args.selective_checkpointing
        )

    # Set model as trainable.
    transformer.train()

    # transformer = transformer.to(dtype=torch.bfloat16)

    if args.model_type == "wan":
        noise_scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        noise_scheduler.set_timesteps(1000, training=True)
    else:
        noise_scheduler = FlowMatchEulerDiscreteScheduler()

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=args.betas,
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    if args.resume_from_lora_checkpoint:
        transformer, optimizer, init_steps = resume_lora_optimizer(
            transformer, args.resume_from_lora_checkpoint, optimizer
        )
    main_print(f"optimizer: {optimizer}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
    sampler = (
        LengthGroupedSampler(
            args.train_batch_size,
            rank=rank,
            world_size=world_size,
            lengths=train_dataset.lengths,
            group_frame=args.group_frame,
            group_resolution=args.group_resolution,
            sp_size=args.sp_size if args.model_type == "wan" else 1
        )
        if (args.group_frame or args.group_resolution)
        else DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size, shuffle=False
        )
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader)
        / args.gradient_accumulation_steps
        * args.sp_size
        / args.train_sp_batch_size
        / world_size
    )
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if rank <= 0:
        project = args.tracker_project_name or "fastvideo"
        wandb.init(project=project, config=args)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Num Epochs = {args.num_train_epochs}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")
        # TODO

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )

    sp_size_with_dataloader = 1 if args.model_type == "cog15B" or args.model_type == "wan" else args.sp_size
    loader = sp_parallel_dataloader_wrapper(
        train_dataloader,
        device,
        args.train_batch_size,
        sp_size_with_dataloader,
        args.train_sp_batch_size,
    )

    step_times = deque(maxlen=100)

    # dist.barrier()
    # todo future
    for i in range(init_steps):
        next(loader)
    for step in range(init_steps + 1, args.max_train_steps + 1):
        start_time = time.time()
        loss, grad_norm = train_one_step(
            transformer,
            args.model_type,
            optimizer,
            lr_scheduler,
            loader,
            noise_scheduler,
            noise_random_generator,
            args.gradient_accumulation_steps,
            args.sp_size,
            args.precondition_outputs,
            args.max_grad_norm,
            args.weighting_scheme,
            args.logit_mean,
            args.logit_std,
            args.mode_scale,
            extra=extra_rtv,now_step=step,args=args
        )

        step_time = time.time() - start_time
        step_times.append(step_time)
        avg_step_time = sum(step_times) / len(step_times)

        progress_bar.set_postfix(
            {
                "loss": f"{loss:.4f}",
                "step_time": f"{step_time:.2f}s",
                "grad_norm": grad_norm,
            }
        )
        progress_bar.update(1)
        if rank <= 0:
            wandb.log(
                {
                    "train_loss": loss,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "step_time": step_time,
                    "avg_step_time": avg_step_time,
                    "grad_norm": grad_norm,
                },
                step=step,
            )
        if step % args.checkpointing_steps == 0:
            if args.use_lora:
                # Save LoRA weights
                save_lora_checkpoint(
                    transformer, optimizer, rank, args.output_dir, step
                )
            else:
                # Your existing checkpoint saving code
                save_checkpoint(transformer, optimizer, rank, args.output_dir, step, save_optimizer=False)
            dist.barrier()
        if args.log_validation and step % args.validation_steps == 0:
            log_validation(args, transformer, device, torch.bfloat16, step)

    if args.use_lora:
        save_lora_checkpoint(
            transformer, optimizer, rank, args.output_dir, args.max_train_steps
        )
    else:
        save_checkpoint(
            transformer, optimizer, rank, args.output_dir, args.max_train_steps
        )

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()

    torch_profiler.stop()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="mochi", help="The type of model to train."
    )
    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
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
    parser.add_argument("--group_frame", action="store_true")  # TODO
    parser.add_argument("--group_resolution", action="store_true")  # TODO

    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--dit_model_cfg", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.1)
    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )

    # validation & logs
    parser.add_argument("--validation_num_frames", type=int, default=163, \
                        help="Number of frames to generate the video when validating.",)
    parser.add_argument("--validation_width", type=int, default=163, \
                        help="Width of frames to generate the video when validating.",)
    parser.add_argument("--validation_height", type=int, default=163, \
                        help="Height of frames to generate the video when validating.",)
    parser.add_argument("--validation_prompt_dir", type=str)
    parser.add_argument("--uncond_prompt_dir", type=str)
    parser.add_argument(
        "--validation_sampling_steps",
        type=str,
        default="64",
        help="use ',' to split multi sampling steps",
    )
    parser.add_argument(
        "--validation_guidance_scale",
        type=str,
        default="4.5",
        help="use ',' to split multi scale",
    )
    parser.add_argument("--validation_steps", type=int, default=50)
    parser.add_argument("--log_validation", action="store_true")
    parser.add_argument("--tracker_project_name", type=str, default=None)
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--resume_from_lora_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous lora checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
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

    # optimizer & scheduler & Training
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="Whether to use LoRA for finetuning.",
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=256, help="Alpha parameter for LoRA."
    )
    parser.add_argument(
        "--lora_rank", type=int, default=128, help="LoRA rank parameter. "
    )
    parser.add_argument("--fsdp_sharding_startegy", default="full")

    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="uniform",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "uniform"],
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply."
    )
    parser.add_argument(
        "--betas", type=float, nargs=2, default=(0.9, 0.999), help="Betas on AdamW optimizer."
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )

    main(parser)
