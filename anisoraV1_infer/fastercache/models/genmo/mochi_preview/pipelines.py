import os
import random
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union, cast

import numpy as np
import ray
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from safetensors.torch import load_file
from torch import nn
from torch.distributed.fsdp import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import (
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from transformers import T5EncoderModel, T5Tokenizer
from transformers.models.t5.modeling_t5 import T5Block

import genmo.mochi_preview.dit.joint_model.context_parallel as cp
import genmo.mochi_preview.vae.cp_conv as cp_conv
from genmo.mochi_preview.vae.model import Decoder, apply_tiled
from genmo.lib.progress import get_new_progress_bar, progress_bar
from genmo.lib.utils import Timer

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention import sdpa_kernel

import genmo.mochi_preview.dit.joint_model.context_parallel as cp
from genmo.mochi_preview.dit.joint_model.layers import (
    FeedForward,
    PatchEmbed,
    RMSNorm,
    TimestepEmbedder,
)
from genmo.mochi_preview.dit.joint_model.mod_rmsnorm import modulated_rmsnorm
from genmo.mochi_preview.dit.joint_model.residual_tanh_gated_rmsnorm import (
    residual_tanh_gated_rmsnorm,
)
from genmo.mochi_preview.dit.joint_model.rope_mixed import (
    compute_mixed_rotation,
    create_position_matrix,
)
from genmo.mochi_preview.dit.joint_model.temporal_rope import apply_rotary_emb_qk_real
from genmo.mochi_preview.dit.joint_model.utils import (
    AttentionPool,
    modulate,
    pad_and_split_xy,
    unify_streams,
)

COMPILE_FINAL_LAYER = os.environ.get("COMPILE_DIT") == "1"
COMPILE_MMDIT_BLOCK = os.environ.get("COMPILE_DIT") == "1"

from genmo.lib.attn_imports import comfy_attn, flash_varlen_qkvpacked_attn, sage_attn, sdpa_attn_ctx



def linear_quadratic_schedule(num_steps, threshold_noise, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return sigma_schedule


T5_MODEL = "google/t5-v1_1-xxl"
MAX_T5_TOKEN_LENGTH = 256


def setup_fsdp_sync(model, device_id, *, param_dtype, auto_wrap_policy) -> FSDP:
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        device_id=device_id,
        sync_module_states=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


class ModelFactory(ABC):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def get_model(self, *, local_rank: int, device_id: Union[int, Literal["cpu"]], world_size: int) -> Any:
        if device_id == "cpu":
            assert world_size == 1, "CPU offload only supports single-GPU inference"


class T5ModelFactory(ModelFactory):
    def __init__(self):
        super().__init__()

    def get_model(self, *, local_rank, device_id, world_size):
        super().get_model(local_rank=local_rank, device_id=device_id, world_size=world_size)
        model = T5EncoderModel.from_pretrained(T5_MODEL)
        if world_size > 1:
            model = setup_fsdp_sync(
                model,
                device_id=device_id,
                param_dtype=torch.float32,
                auto_wrap_policy=partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={
                        T5Block,
                    },
                ),
            )
        elif isinstance(device_id, int):
            model = model.to(torch.device(f"cuda:{device_id}"))  # type: ignore
        return model.eval()


class DitModelFactory(ModelFactory):
    def __init__(self, *, model_path: str, model_dtype: str, attention_mode: Optional[str] = None):
        if attention_mode is None:
            from genmo.lib.attn_imports import flash_varlen_qkvpacked_attn # type: ignore

            attention_mode = "sdpa" if flash_varlen_qkvpacked_attn is None else "flash"
        print(f"Attention mode: {attention_mode}")
        super().__init__(model_path=model_path, model_dtype=model_dtype, attention_mode=attention_mode)

    def get_model(self, *, local_rank, device_id, world_size):
        # TODO(ved): Set flag for torch.compile
        from genmo.mochi_preview.dit.joint_model.asymm_models_joint import (
            AsymmDiTJoint,
        )

        model: nn.Module = torch.nn.utils.skip_init(
            AsymmDiTJoint,
            depth=48,
            patch_size=2,
            num_heads=24,
            hidden_size_x=3072,
            hidden_size_y=1536,
            mlp_ratio_x=4.0,
            mlp_ratio_y=4.0,
            in_channels=12,
            qk_norm=True,
            qkv_bias=False,
            out_bias=True,
            patch_embed_bias=True,
            timestep_mlp_bias=True,
            timestep_scale=1000.0,
            t5_feat_dim=4096,
            t5_token_length=256,
            rope_theta=10000.0,
            attention_mode=self.kwargs["attention_mode"],
        )

        if local_rank == 0:
            # FSDP syncs weights from rank 0 to all other ranks
            model.load_state_dict(load_file(self.kwargs["model_path"]))

        if world_size > 1:
            assert self.kwargs["model_dtype"] == "bf16", "FP8 is not supported for multi-GPU inference"
            model = setup_fsdp_sync(
                model,
                device_id=device_id,
                param_dtype=torch.bfloat16,
                auto_wrap_policy=partial(
                    lambda_auto_wrap_policy,
                    lambda_fn=lambda m: m in model.blocks,
                ),
            )
        elif isinstance(device_id, int):
            model = model.to(torch.device(f"cuda:{device_id}"))
        return model.eval()


class DecoderModelFactory(ModelFactory):
    def __init__(self, *, model_path: str, model_stats_path: str):
        super().__init__(model_path=model_path, model_stats_path=model_stats_path)

    def get_model(self, *, local_rank, device_id, world_size):
        # TODO(ved): Set flag for torch.compile
        # TODO(ved): Use skip_init
        import json

        decoder = Decoder(
            out_channels=3,
            base_channels=128,
            channel_multipliers=[1, 2, 4, 6],
            temporal_expansions=[1, 2, 3],
            spatial_expansions=[2, 2, 2],
            num_res_blocks=[3, 3, 4, 6, 3],
            latent_dim=12,
            has_attention=[False, False, False, False, False],
            padding_mode="replicate",
            output_norm=False,
            nonlinearity="silu",
            output_nonlinearity="silu",
            causal=True,
        )
        # VAE is not FSDP-wrapped
        state_dict = load_file(self.kwargs["model_path"])
        decoder.load_state_dict(state_dict, strict=True)
        device = torch.device(f"cuda:{device_id}") if isinstance(device_id, int) else "cpu"
        decoder.eval().to(device)
        vae_stats = json.load(open(self.kwargs["model_stats_path"]))
        decoder.register_buffer("vae_mean", torch.tensor(vae_stats["mean"], device=device))
        decoder.register_buffer("vae_std", torch.tensor(vae_stats["std"], device=device))
        return decoder


def get_conditioning(tokenizer, encoder, device, batch_inputs, *, prompt: str, negative_prompt: str):
    if batch_inputs:
        return dict(batched=get_conditioning_for_prompts(tokenizer, encoder, device, [prompt, negative_prompt]))
    else:
        cond_input = get_conditioning_for_prompts(tokenizer, encoder, device, [prompt])
        null_input = get_conditioning_for_prompts(tokenizer, encoder, device, [negative_prompt])
        return dict(cond=cond_input, null=null_input)


def get_conditioning_for_prompts(tokenizer, encoder, device, prompts: List[str]):
    assert len(prompts) in [1, 2]  # [neg] or [pos] or [pos, neg]
    B = len(prompts)
    t5_toks = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=MAX_T5_TOKEN_LENGTH,
        return_tensors="pt",
        return_attention_mask=True,
    )
    caption_input_ids_t5 = t5_toks["input_ids"]
    caption_attention_mask_t5 = t5_toks["attention_mask"].bool()
    del t5_toks

    assert caption_input_ids_t5.shape == (B, MAX_T5_TOKEN_LENGTH)
    assert caption_attention_mask_t5.shape == (B, MAX_T5_TOKEN_LENGTH)

    # Special-case empty negative prompt by zero-ing it
    if prompts[-1] == "":
        caption_input_ids_t5[-1] = 0
        caption_attention_mask_t5[-1] = False

    caption_input_ids_t5 = caption_input_ids_t5.to(device, non_blocking=True)
    caption_attention_mask_t5 = caption_attention_mask_t5.to(device, non_blocking=True)

    y_mask = [caption_attention_mask_t5]
    y_feat = [encoder(caption_input_ids_t5, caption_attention_mask_t5).last_hidden_state.detach()]
    # Sometimes returns a tensor, othertimes a tuple, not sure why
    # See: https://huggingface.co/genmo/mochi-1-preview/discussions/3
    assert tuple(y_feat[-1].shape) == (B, MAX_T5_TOKEN_LENGTH, 4096)
    assert y_feat[-1].dtype == torch.float32

    return dict(y_mask=y_mask, y_feat=y_feat)


def compute_packed_indices(
    device: torch.device, text_mask: torch.Tensor, num_latents: int
) -> Dict[str, Union[torch.Tensor, int]]:
    """
    Based on https://github.com/Dao-AILab/flash-attention/blob/765741c1eeb86c96ee71a3291ad6968cfbf4e4a1/flash_attn/bert_padding.py#L60-L80

    Args:
        num_latents: Number of latent tokens
        text_mask: (B, L) List of boolean tensor indicating which text tokens are not padding.

    Returns:
        packed_indices: Dict with keys for Flash Attention:
            - valid_token_indices_kv: up to (B * (N + L),) tensor of valid token indices (non-padding)
                                   in the packed sequence.
            - cu_seqlens_kv: (B + 1,) tensor of cumulative sequence lengths in the packed sequence.
            - max_seqlen_in_batch_kv: int of the maximum sequence length in the batch.
    """
    # Create an expanded token mask saying which tokens are valid across both visual and text tokens.
    PATCH_SIZE = 2
    num_visual_tokens = num_latents // (PATCH_SIZE**2)
    assert num_visual_tokens > 0

    mask = F.pad(text_mask, (num_visual_tokens, 0), value=True)  # (B, N + L)
    seqlens_in_batch = mask.sum(dim=-1, dtype=torch.int32)  # (B,)
    valid_token_indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()  # up to (B * (N + L),)
    assert valid_token_indices.size(0) >= text_mask.size(0) * num_visual_tokens  # At least (B * N,)
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen_in_batch = seqlens_in_batch.max().item()

    return {
        "cu_seqlens_kv": cu_seqlens.to(device, non_blocking=True),
        "max_seqlen_in_batch_kv": cast(int, max_seqlen_in_batch),
        "valid_token_indices_kv": valid_token_indices.to(device, non_blocking=True),
    }


def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"


def fastercache_asynmmetricjb_forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        y: torch.Tensor,
        cnt=None,
        bid=0,
        cond=False,
        **attn_kwargs,
    ):
        N = x.size(1)

        c = F.silu(c)
        mod_x = self.mod_x(c)
        scale_msa_x, gate_msa_x, scale_mlp_x, gate_mlp_x = mod_x.chunk(4, dim=1)

        mod_y = self.mod_y(c)
        if self.update_y:
            scale_msa_y, gate_msa_y, scale_mlp_y, gate_mlp_y = mod_y.chunk(4, dim=1)
        else:
            scale_msa_y = mod_y
        
        if cnt >= 20 and cnt % 2 == 0:
            x_attn = self.x_attn_cache[1] + (self.x_attn_cache[1] - self.x_attn_cache[0])*max(0.5, 1-(63-cnt)/63)
            y_attn = self.y_attn_cache
        else:
            x_attn, y_attn = self.attn(
                x,
                y,
                scale_x=scale_msa_x,
                scale_y=scale_msa_y,
                **attn_kwargs,
            )
            if cnt==0:
                self.x_attn_cache = None
                self.y_attn_cache = None
            elif cnt==5:
                self.x_attn_cache = [x_attn,x_attn]
                self.y_attn_cache = y_attn
            elif cnt>5:
                self.x_attn_cache = [self.x_attn_cache[1], x_attn]
                self.y_attn_cache = y_attn
        
        assert x_attn.size(1) == N
        x = residual_tanh_gated_rmsnorm(x, x_attn, gate_msa_x)
        if self.update_y:
            y = residual_tanh_gated_rmsnorm(y, y_attn, gate_msa_y)

        # MLP block.
        x = self.ff_block_x(x, scale_mlp_x, gate_mlp_x)
        if self.update_y:
            y = self.ff_block_y(y, scale_mlp_y, gate_mlp_y)

        return x, y


def fastercache_asynmmetridit_forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        y_feat: List[torch.Tensor],
        y_mask: List[torch.Tensor],
        packed_indices: Dict[str, torch.Tensor] = None,
        rope_cos: torch.Tensor = None,
        rope_sin: torch.Tensor = None,
        cnt=None,
        cond=False,
    ):
        B, _, T, H, W = x.shape

        # Use EFFICIENT_ATTENTION backend for T5 pooling, since we have a mask.
        # Have to call sdpa_kernel outside of a torch.compile region.
        with sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
            x, c, y_feat, rope_cos, rope_sin = self.prepare(x, sigma, y_feat[0], y_mask[0])
        del y_mask

        cp_rank, cp_size = cp.get_cp_rank_size()
        N = x.size(1)
        M = N // cp_size
        assert N % cp_size == 0, f"Visual sequence length ({x.shape[1]}) must be divisible by cp_size ({cp_size})."

        if cp_size > 1:
            x = x.narrow(1, cp_rank * M, M)

            assert self.num_heads % cp_size == 0
            local_heads = self.num_heads // cp_size
            rope_cos = rope_cos.narrow(1, cp_rank * local_heads, local_heads)
            rope_sin = rope_sin.narrow(1, cp_rank * local_heads, local_heads)

        for i, block in enumerate(self.blocks):
            x, y_feat = block(
                x,
                c,
                y_feat,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                packed_indices=packed_indices,
                cnt=cnt,
                bid=i,
                cond=cond,
            )  # (B, M, D), (B, L, D)
        del y_feat  # Final layers don't use dense text features.

        x = self.final_layer(x, c)  # (B, M, patch_size ** 2 * out_channels)

        patch = x.size(2)
        x = cp.all_gather(x)
        x = rearrange(x, "(G B) M P -> B (G M) P", G=cp_size, P=patch)
        x = rearrange(
            x,
            "B (T hp wp) (p1 p2 c) -> B c T (hp p1) (wp p2)",
            T=T,
            hp=H // self.patch_size,
            wp=W // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.out_channels,
        )

        return x


import torch.fft
@torch.no_grad()
def fft(tensor):
    tensor_fft = torch.fft.fft2(tensor)
    tensor_fft_shifted = torch.fft.fftshift(tensor_fft)
    B, C, H, W = tensor.size()
    radius = min(H, W) // 5  
            
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W))
    center_x, center_y = W // 2, H // 2
    mask = (X - center_x) ** 2 + (Y - center_y) ** 2 <= radius ** 2
    low_freq_mask = mask.unsqueeze(0).unsqueeze(0).to(tensor.device)
    high_freq_mask = ~low_freq_mask
            
    low_freq_fft = tensor_fft_shifted * low_freq_mask
    high_freq_fft = tensor_fft_shifted * high_freq_mask
    return low_freq_fft, high_freq_fft

cfg_delta_low = None
cfg_delta_high = None

def sample_model_fastercache(device, dit, conditioning, **args):
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    generator = torch.Generator(device=device)
    generator.manual_seed(args["seed"])

    w, h, t = args["width"], args["height"], args["num_frames"]
    sample_steps = args["num_inference_steps"]
    cfg_schedule = args["cfg_schedule"]
    sigma_schedule = args["sigma_schedule"]

    assert_eq(len(cfg_schedule), sample_steps, "cfg_schedule must have length sample_steps")
    assert_eq((t - 1) % 6, 0, "t - 1 must be divisible by 6")
    assert_eq(
        len(sigma_schedule),
        sample_steps + 1,
        "sigma_schedule must have length sample_steps + 1",
    )

    B = 1
    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 6
    IN_CHANNELS = 12
    latent_t = ((t - 1) // TEMPORAL_DOWNSAMPLE) + 1
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    z = torch.randn(
        (B, IN_CHANNELS, latent_t, latent_h, latent_w),
        device=device,
        dtype=torch.float32,
    )

    num_latents = latent_t * latent_h * latent_w
    cond_batched = cond_text = cond_null = None
    if "cond" in conditioning:
        cond_text = conditioning["cond"]
        cond_null = conditioning["null"]
        cond_text["packed_indices"] = compute_packed_indices(device, cond_text["y_mask"][0], num_latents)
        cond_null["packed_indices"] = compute_packed_indices(device, cond_null["y_mask"][0], num_latents)
    else:
        cond_batched = conditioning["batched"]
        cond_batched["packed_indices"] = compute_packed_indices(device, cond_batched["y_mask"][0], num_latents)
        z = repeat(z, "b ... -> (repeat b) ...", repeat=2)

    for _name, _module in dit.named_modules():
        if _module.__class__.__name__=='AsymmetricJointBlock':
            _module.__class__.forward  = fastercache_asynmmetricjb_forward
        
        if _module.__class__.__name__=='AsymmDiTJoint':
            _module.__class__.forward  = fastercache_asynmmetridit_forward

    def model_fn(*, z, sigma, cfg_scale, cnt=None):
        if cond_batched:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = dit(z, sigma, **cond_batched)
            out_cond, out_uncond = torch.chunk(out, chunks=2, dim=0)
        else:
            nonlocal cond_text, cond_null
            global cfg_delta_low, cfg_delta_high 
            if cnt>=22 and cnt%5!=0 and cnt!=63:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out_cond = dit(z, sigma, cnt=cnt, cond=True, **cond_text)
                    (bb, cc, tt, hh, ww) = out_cond.shape
                    out_cond_forfft = rearrange(out_cond, "B C T H W -> (B T) C H W", B=bb, C=cc, T=tt, H=hh, W=ww)
                    lf_c, hf_c = fft(out_cond_forfft.float())
                    if cnt >= 30:
                        cfg_delta_high = cfg_delta_high * 1.25
                    if cnt <= 50:
                        cfg_delta_low = cfg_delta_low * 1.15
                    new_hf_uc = cfg_delta_high + hf_c
                    new_lf_uc = cfg_delta_low + lf_c
                    combine_uc = new_lf_uc + new_hf_uc
                    combined_fft = torch.fft.ifftshift(combine_uc)
                    recovered_uncond = torch.fft.ifft2(combined_fft).real
                    out_uncond = rearrange(recovered_uncond, "(B T) C H W -> B C T H W", B=bb, C=cc, T=tt, H=hh, W=ww)

            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out_cond = dit(z, sigma, cnt=cnt, cond=True, **cond_text)
                    out_uncond = dit(z, sigma, cnt=cnt, cond=False, **cond_null)
                    if cnt>=20:
                        (bb, cc, tt, hh, ww) = out_cond.shape
                        out_cond_forfft = rearrange(out_cond, "B C T H W -> (B T) C H W", B=bb, C=cc, T=tt, H=hh, W=ww)
                        out_uncond_forfft = rearrange(out_uncond, "B C T H W -> (B T) C H W", B=bb, C=cc, T=tt, H=hh, W=ww)

                        lf_c, hf_c = fft(out_cond_forfft.float())
                        lf_uc, hf_uc = fft(out_uncond_forfft.float())

                        cfg_delta_high = hf_uc - hf_c
                        cfg_delta_low = lf_uc - lf_c

        assert out_cond.shape == out_uncond.shape
        return out_uncond + cfg_scale * (out_cond - out_uncond), out_cond

    for i in get_new_progress_bar(range(0, sample_steps), desc="Sampling"):
        sigma = sigma_schedule[i]
        dsigma = sigma - sigma_schedule[i + 1]

        # `pred` estimates `z_0 - eps`.
        pred, output_cond = model_fn(
            z=z,
            sigma=torch.full([B] if cond_text else [B * 2], sigma, device=z.device),
            cfg_scale=cfg_schedule[i],
            cnt=i,
        )
        pred = pred.to(z)
        output_cond = output_cond.to(z)
        z = z + dsigma * pred

    return z[:B] if cond_batched else z


def sample_model(device, dit, conditioning, **args):
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    generator = torch.Generator(device=device)
    generator.manual_seed(args["seed"])

    w, h, t = args["width"], args["height"], args["num_frames"]
    sample_steps = args["num_inference_steps"]
    cfg_schedule = args["cfg_schedule"]
    sigma_schedule = args["sigma_schedule"]

    assert_eq(len(cfg_schedule), sample_steps, "cfg_schedule must have length sample_steps")
    assert_eq((t - 1) % 6, 0, "t - 1 must be divisible by 6")
    assert_eq(
        len(sigma_schedule),
        sample_steps + 1,
        "sigma_schedule must have length sample_steps + 1",
    )

    B = 1
    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 6
    IN_CHANNELS = 12
    latent_t = ((t - 1) // TEMPORAL_DOWNSAMPLE) + 1
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    z = torch.randn(
        (B, IN_CHANNELS, latent_t, latent_h, latent_w),
        device=device,
        dtype=torch.float32,
    )

    num_latents = latent_t * latent_h * latent_w
    cond_batched = cond_text = cond_null = None
    if "cond" in conditioning:
        cond_text = conditioning["cond"]
        cond_null = conditioning["null"]
        cond_text["packed_indices"] = compute_packed_indices(device, cond_text["y_mask"][0], num_latents)
        cond_null["packed_indices"] = compute_packed_indices(device, cond_null["y_mask"][0], num_latents)
    else:
        cond_batched = conditioning["batched"]
        cond_batched["packed_indices"] = compute_packed_indices(device, cond_batched["y_mask"][0], num_latents)
        z = repeat(z, "b ... -> (repeat b) ...", repeat=2)

    def model_fn(*, z, sigma, cfg_scale):
        if cond_batched:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = dit(z, sigma, **cond_batched)
            out_cond, out_uncond = torch.chunk(out, chunks=2, dim=0)
        else:
            nonlocal cond_text, cond_null
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out_cond = dit(z, sigma, **cond_text)
                out_uncond = dit(z, sigma, **cond_null)
        assert out_cond.shape == out_uncond.shape
        return out_uncond + cfg_scale * (out_cond - out_uncond), out_cond

    for i in get_new_progress_bar(range(0, sample_steps), desc="Sampling"):
        sigma = sigma_schedule[i]
        dsigma = sigma - sigma_schedule[i + 1]

        # `pred` estimates `z_0 - eps`.
        pred, output_cond = model_fn(
            z=z,
            sigma=torch.full([B] if cond_text else [B * 2], sigma, device=z.device),
            cfg_scale=cfg_schedule[i],
        )
        pred = pred.to(z)
        output_cond = output_cond.to(z)
        z = z + dsigma * pred

    return z[:B] if cond_batched else z


def decoded_latents_to_frames(samples):
    samples = samples.float()
    samples = (samples + 1.0) / 2.0
    samples.clamp_(0.0, 1.0)
    frames = rearrange(samples, "b c t h w -> b t h w c")
    return frames


def decode_latents(decoder, z):
    cp_rank, cp_size = cp.get_cp_rank_size()
    z = z.tensor_split(cp_size, dim=2)[cp_rank]  # split along temporal dim
    with torch.autocast("cuda", dtype=torch.bfloat16):
        samples = decoder(z)
    samples = cp_conv.gather_all_frames(samples)
    return decoded_latents_to_frames(samples)

def decode_latents_pure(decoder, z):
    # cp_rank, cp_size = cp.get_cp_rank_size()
    # z = z.tensor_split(cp_size, dim=2)[cp_rank]  # split along temporal dim
    with torch.autocast("cuda", dtype=torch.bfloat16):
        samples = decoder(z)
    # samples = cp_conv.gather_all_frames(samples)
    return decoded_latents_to_frames(samples)

@torch.inference_mode()
def decode_latents_tiled_full(
    decoder,
    z,
    *,
    tile_sample_min_height: int = 240,
    tile_sample_min_width: int = 424,
    tile_overlap_factor_height: float = 0.1666,
    tile_overlap_factor_width: float = 0.2,
    auto_tile_size: bool = True,
    frame_batch_size: int = 6,
):
    B, C, T, H, W = z.shape
    assert frame_batch_size <= T, f"frame_batch_size must be <= T, got {frame_batch_size} > {T}"

    tile_sample_min_height = tile_sample_min_height if not auto_tile_size else H // 2 * 8
    tile_sample_min_width = tile_sample_min_width if not auto_tile_size else W // 2 * 8

    tile_latent_min_height = int(tile_sample_min_height / 8)
    tile_latent_min_width = int(tile_sample_min_width / 8)

    def blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    overlap_height = int(tile_latent_min_height * (1 - tile_overlap_factor_height))
    overlap_width = int(tile_latent_min_width * (1 - tile_overlap_factor_width))
    blend_extent_height = int(tile_sample_min_height * tile_overlap_factor_height)
    blend_extent_width = int(tile_sample_min_width * tile_overlap_factor_width)
    row_limit_height = tile_sample_min_height - blend_extent_height
    row_limit_width = tile_sample_min_width - blend_extent_width

    # Split z into overlapping tiles and decode them separately.
    # The tiles have an overlap to avoid seams between tiles.
    pbar = get_new_progress_bar(
        desc="Decoding latent tiles",
        total=len(range(0, H, overlap_height)) * len(range(0, W, overlap_width)) * len(range(T // frame_batch_size)),
    )
    rows = []
    for i in range(0, H, overlap_height):
        row = []
        for j in range(0, W, overlap_width):
            temporal = []
            for k in range(T // frame_batch_size):
                remaining_frames = T % frame_batch_size
                start_frame = frame_batch_size * k + (0 if k == 0 else remaining_frames)
                end_frame = frame_batch_size * (k + 1) + remaining_frames
                tile = z[
                    :,
                    :,
                    start_frame:end_frame,
                    i : i + tile_latent_min_height,
                    j : j + tile_latent_min_width,
                ]
                tile = decoder(tile)
                temporal.append(tile)
                pbar.update(1)
            row.append(torch.cat(temporal, dim=2))
        rows.append(row)

    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            # blend the above tile and the left tile
            # to the current tile and add the current tile to the result row
            if i > 0:
                tile = blend_v(rows[i - 1][j], tile, blend_extent_height)
            if j > 0:
                tile = blend_h(row[j - 1], tile, blend_extent_width)
            result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
        result_rows.append(torch.cat(result_row, dim=4))

    return decoded_latents_to_frames(torch.cat(result_rows, dim=3))

@torch.inference_mode()
def decode_latents_tiled_spatial(
    decoder,
    z,
    *,
    num_tiles_w: int,
    num_tiles_h: int,
    overlap: int = 0,  # Number of pixel of overlap between adjacent tiles.
    # Use a factor of 2 times the latent downsample factor.
    min_block_size: int = 1,  # Minimum number of pixels in each dimension when subdividing.
):
    decoded = apply_tiled(decoder, z, num_tiles_w, num_tiles_h, overlap, min_block_size)
    assert decoded is not None, f"Failed to decode latents with tiled spatial method"
    return decoded

@contextmanager
def move_to_device(model: nn.Module, target_device):
    og_device = next(model.parameters()).device
    if og_device == target_device:
        print(f"move_to_device is a no-op model is already on {target_device}")
    else:
        print(f"moving model from {og_device} -> {target_device}")

    model.to(target_device)
    yield
    if og_device != target_device:
        print(f"moving model from {target_device} -> {og_device}")
    model.to(og_device)


def t5_tokenizer():
    return T5Tokenizer.from_pretrained(T5_MODEL, legacy=False)


class MochiSingleGPUPipeline:
    def __init__(
        self,
        *,
        text_encoder_factory: ModelFactory,
        dit_factory: ModelFactory,
        decoder_factory: ModelFactory,
        cpu_offload: Optional[bool] = False,
        decode_type: str = "full",
        decode_args: Optional[Dict[str, Any]] = None,
    ):
        self.device = torch.device("cuda:0")
        self.tokenizer = t5_tokenizer()
        t = Timer()
        self.cpu_offload = cpu_offload
        self.decode_args = decode_args or {}
        self.decode_type = decode_type
        init_id = "cpu" if cpu_offload else 0
        with t("load_text_encoder"):
            self.text_encoder = text_encoder_factory.get_model(
                local_rank=0,
                device_id=init_id,
                world_size=1,
            )
        with t("load_dit"):
            self.dit = dit_factory.get_model(local_rank=0, device_id=init_id, world_size=1)
        with t("load_vae"):
            self.decoder = decoder_factory.get_model(local_rank=0, device_id=init_id, world_size=1)
        t.print_stats()

    def __call__(self, batch_cfg, prompt, negative_prompt, **kwargs):
        # progress_bar(type="tqdm"),
        with torch.inference_mode():
            print_max_memory = lambda: print(
                f"Max memory reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB"
            )
            print_max_memory()
            with move_to_device(self.text_encoder, self.device):
                conditioning = get_conditioning(
                    self.tokenizer,
                    self.text_encoder,
                    self.device,
                    batch_cfg,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                )
            print_max_memory()
            with move_to_device(self.dit, self.device):
                if kwargs['acc']==False:
                    latents = sample_model(self.device, self.dit, conditioning, **kwargs)
                else:
                    latents = sample_model_fastercache(self.device, self.dit, conditioning, **kwargs)
            print_max_memory()
            with move_to_device(self.decoder, self.device):
                frames = (
                    decode_latents_tiled_full(self.decoder, latents, **self.decode_args)
                    if self.decode_type == "tiled_full"
                    else
                    decode_latents_tiled_spatial(self.decoder, latents, **self.decode_args)
                    if self.decode_type == "tiled_spatial"
                    else decode_latents_pure(self.decoder, latents)
                )
            print_max_memory()
            return frames.cpu().numpy()


### ALL CODE BELOW HERE IS FOR MULTI-GPU MODE ###


# In multi-gpu mode, all models must belong to a device which has a predefined context parallel group
# So it doesn't make sense to work with models individually
class MultiGPUContext:
    def __init__(
        self,
        *,
        text_encoder_factory,
        dit_factory,
        decoder_factory,
        device_id,
        local_rank,
        world_size,
    ):
        t = Timer()
        self.device = torch.device(f"cuda:{device_id}")
        print(f"Initializing rank {local_rank+1}/{world_size}")
        assert world_size > 1, f"Multi-GPU mode requires world_size > 1, got {world_size}"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        with t("init_process_group"):
            dist.init_process_group(
                "nccl",
                rank=local_rank,
                world_size=world_size,
                device_id=self.device,  # force non-lazy init
            )
        pg = dist.group.WORLD
        cp.set_cp_group(pg, list(range(world_size)), local_rank)
        distributed_kwargs = dict(local_rank=local_rank, device_id=device_id, world_size=world_size)
        self.world_size = world_size
        self.tokenizer = t5_tokenizer()
        with t("load_text_encoder"):
            self.text_encoder = text_encoder_factory.get_model(**distributed_kwargs)
        with t("load_dit"):
            self.dit = dit_factory.get_model(**distributed_kwargs)
        with t("load_vae"):
            self.decoder = decoder_factory.get_model(**distributed_kwargs)
        self.local_rank = local_rank
        t.print_stats()

    def run(self, *, fn, **kwargs):
        return fn(self, **kwargs)


class MochiMultiGPUPipeline:
    def __init__(
        self,
        *,
        text_encoder_factory: ModelFactory,
        dit_factory: ModelFactory,
        decoder_factory: ModelFactory,
        world_size: int,
    ):
        ray.init()
        RemoteClass = ray.remote(MultiGPUContext)
        self.ctxs = [
            RemoteClass.options(num_gpus=1).remote(
                text_encoder_factory=text_encoder_factory,
                dit_factory=dit_factory,
                decoder_factory=decoder_factory,
                world_size=world_size,
                device_id=0,
                local_rank=i,
            )
            for i in range(world_size)
        ]
        for ctx in self.ctxs:
            ray.get(ctx.__ray_ready__.remote())

    def __call__(self, **kwargs):
        def sample(ctx, *, batch_cfg, prompt, negative_prompt, **kwargs):
            with progress_bar(type="ray_tqdm", enabled=ctx.local_rank == 0), torch.inference_mode():
                conditioning = get_conditioning(
                    ctx.tokenizer,
                    ctx.text_encoder,
                    ctx.device,
                    batch_cfg,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                )
                if kwargs['acc']==False:
                    latents = sample_model(ctx.device, ctx.dit, conditioning=conditioning, **kwargs)
                else:
                    latents = sample_model_fastercache(ctx.device, ctx.dit, conditioning=conditioning, **kwargs)
                if ctx.local_rank == 0:
                    torch.save(latents, "latents.pt")
                frames = decode_latents(ctx.decoder, latents)
            return frames.cpu().numpy()

        return ray.get([ctx.run.remote(fn=sample, **kwargs, show_progress=i == 0) for i, ctx in enumerate(self.ctxs)])[
            0
        ]
