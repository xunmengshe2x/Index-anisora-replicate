"""
This script demonstrates how to convert and generate video from a text prompt using CogVideoX with 🤗Huggingface Diffusers Pipeline.

Note:
    This script requires the `diffusers>=0.30.0` library to be installed.

Run the script:
    $ python convert_and_generate.py --transformer_ckpt_path <path_to_transformer_checkpoint> --vae_ckpt_path <path_to_vae_checkpoint> --output_path <path_to_output_directory> --text_encoder_path <path_to_t5>

Functions:
    - reassign_query_key_value_inplace: Reassigns the query, key, and value weights in-place.
    - reassign_query_key_layernorm_inplace: Reassigns layer normalization for query and key in-place.
    - reassign_adaln_norm_inplace: Reassigns adaptive layer normalization in-place.
    - remove_keys_inplace: Removes specified keys from the state_dict in-place.
    - replace_up_keys_inplace: Replaces keys in the "up" block in-place.
    - get_state_dict: Extracts the state_dict from a saved checkpoint.
    - update_state_dict_inplace: Updates the state_dict with new key assignments in-place.
    - convert_transformer: Converts a transformer checkpoint to the CogVideoX format.
    - convert_vae: Converts a VAE checkpoint to the CogVideoX format.
    - get_args: Parses command-line arguments for the script.
    - generate_video: Generates a video from a text prompt using the CogVideoX pipeline.
"""

import argparse
from typing import Any, Dict

import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler, CogVideoXPipeline, CogVideoXTransformer3DModel
from transformers import T5EncoderModel, T5Tokenizer


# Function to reassign the query, key, and value weights in-place
def reassign_query_key_value_inplace(key: str, state_dict: Dict[str, Any]):
    to_q_key = key.replace("query_key_value", "to_q")
    to_k_key = key.replace("query_key_value", "to_k")
    to_v_key = key.replace("query_key_value", "to_v")
    to_q, to_k, to_v = torch.chunk(state_dict[key], chunks=3, dim=0)
    state_dict[to_q_key] = to_q
    state_dict[to_k_key] = to_k
    state_dict[to_v_key] = to_v
    state_dict.pop(key)


# Function to reassign layer normalization for query and key in-place
def reassign_query_key_layernorm_inplace(key: str, state_dict: Dict[str, Any]):
    layer_id, weight_or_bias = key.split(".")[-2:]

    if "query" in key:
        new_key = f"transformer_blocks.{layer_id}.attn1.norm_q.{weight_or_bias}"
    elif "key" in key:
        new_key = f"transformer_blocks.{layer_id}.attn1.norm_k.{weight_or_bias}"

    state_dict[new_key] = state_dict.pop(key)


# Function to reassign adaptive layer normalization in-place
def reassign_adaln_norm_inplace(key: str, state_dict: Dict[str, Any]):
    layer_id, _, weight_or_bias = key.split(".")[-3:]

    weights_or_biases = state_dict[key].chunk(12, dim=0)
    norm1_weights_or_biases = torch.cat(weights_or_biases[0:3] + weights_or_biases[6:9])
    norm2_weights_or_biases = torch.cat(weights_or_biases[3:6] + weights_or_biases[9:12])

    norm1_key = f"transformer_blocks.{layer_id}.norm1.linear.{weight_or_bias}"
    state_dict[norm1_key] = norm1_weights_or_biases

    norm2_key = f"transformer_blocks.{layer_id}.norm2.linear.{weight_or_bias}"
    state_dict[norm2_key] = norm2_weights_or_biases

    state_dict.pop(key)


# Function to remove keys from state_dict in-place
def remove_keys_inplace(key: str, state_dict: Dict[str, Any]):
    state_dict.pop(key)


# Function to replace keys in the "up" block in-place
def replace_up_keys_inplace(key: str, state_dict: Dict[str, Any]):
    key_split = key.split(".")
    layer_index = int(key_split[2])
    replace_layer_index = 4 - 1 - layer_index

    key_split[1] = "up_blocks"
    key_split[2] = str(replace_layer_index)
    new_key = ".".join(key_split)

    state_dict[new_key] = state_dict.pop(key)


# Dictionary for renaming transformer keys
TRANSFORMER_KEYS_RENAME_DICT = {
    "transformer.final_layernorm": "norm_final",
    "transformer": "transformer_blocks",
    "attention": "attn1",
    "mlp": "ff.net",
    "dense_h_to_4h": "0.proj",
    "dense_4h_to_h": "2",
    ".layers": "",
    "dense": "to_out.0",
    "input_layernorm": "norm1.norm",
    "post_attn1_layernorm": "norm2.norm",
    "time_embed.0": "time_embedding.linear_1",
    "time_embed.2": "time_embedding.linear_2",
    "mixins.patch_embed": "patch_embed",
    "mixins.final_layer.norm_final": "norm_out.norm",
    "mixins.final_layer.linear": "proj_out",
    "mixins.final_layer.adaLN_modulation.1": "norm_out.linear",
}

# Dictionary for handling special keys in transformer
TRANSFORMER_SPECIAL_KEYS_REMAP = {
    "query_key_value": reassign_query_key_value_inplace,
    "query_layernorm_list": reassign_query_key_layernorm_inplace,
    "key_layernorm_list": reassign_query_key_layernorm_inplace,
    "adaln_layer.adaLN_modulations": reassign_adaln_norm_inplace,
    "embed_tokens": remove_keys_inplace,
}

# Dictionary for renaming VAE keys
VAE_KEYS_RENAME_DICT = {
    "block.": "resnets.",
    "down.": "down_blocks.",
    "downsample": "downsamplers.0",
    "upsample": "upsamplers.0",
    "nin_shortcut": "conv_shortcut",
    "encoder.mid.block_1": "encoder.mid_block.resnets.0",
    "encoder.mid.block_2": "encoder.mid_block.resnets.1",
    "decoder.mid.block_1": "decoder.mid_block.resnets.0",
    "decoder.mid.block_2": "decoder.mid_block.resnets.1",
}

# Dictionary for handling special keys in VAE
VAE_SPECIAL_KEYS_REMAP = {
    "loss": remove_keys_inplace,
    "up.": replace_up_keys_inplace,
}

# Maximum length of the tokenizer (Must be 226)
TOKENIZER_MAX_LENGTH = 226


# Function to extract the state_dict from a saved checkpoint
def get_state_dict(saved_dict: Dict[str, Any]) -> Dict[str, Any]:
    state_dict = saved_dict
    if "model" in saved_dict.keys():
        state_dict = state_dict["model"]
    if "module" in saved_dict.keys():
        state_dict = state_dict["module"]
    if "state_dict" in saved_dict.keys():
        state_dict = state_dict["state_dict"]
    return state_dict


# Function to update the state_dict with new key assignments in-place
def update_state_dict_inplace(state_dict: Dict[str, Any], old_key: str, new_key: str) -> Dict[str, Any]:
    state_dict[new_key] = state_dict.pop(old_key)


# Function to convert a transformer checkpoint to the CogVideoX format
def convert_transformer(ckpt_path: str):
    PREFIX_KEY = "model.diffusion_model."

    original_state_dict = get_state_dict(torch.load(ckpt_path, map_location="cpu", mmap=True))
    transformer = CogVideoXTransformer3DModel()

    for key in list(original_state_dict.keys()):
        new_key = key[len(PREFIX_KEY) :]
        for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_inplace(original_state_dict, key, new_key)

    for key in list(original_state_dict.keys()):
        for special_key, handler_fn_inplace in TRANSFORMER_SPECIAL_KEYS_REMAP.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, original_state_dict)

    transformer.load_state_dict(original_state_dict, strict=True)
    return transformer


# Function to convert a VAE checkpoint to the CogVideoX format
def convert_vae(ckpt_path: str):
    original_state_dict = get_state_dict(torch.load(ckpt_path, map_location="cpu", mmap=True))
    vae = AutoencoderKLCogVideoX()

    for key in list(original_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in VAE_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_inplace(original_state_dict, key, new_key)

    for key in list(original_state_dict.keys()):
        for special_key, handler_fn_inplace in VAE_SPECIAL_KEYS_REMAP.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, original_state_dict)

    vae.load_state_dict(original_state_dict, strict=True)
    return vae


# Function to parse command-line arguments for the script
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformer_ckpt_path", type=str, default=None, help="Path to original transformer checkpoint"
    )
    parser.add_argument("--vae_ckpt_path", type=str, default=None, help="Path to original vae checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Path where converted model should be saved")
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        required=True,
        default="google/t5-v1_1-xxl",
        help="Path where converted model should be saved",
    )
    parser.add_argument(
        "--text_encoder_cache_dir",
        type=str,
        default=None,
        help="Path to text encoder cache directory. Not needed if text_encoder_path is in your local.",
    )
    parser.add_argument("--fp16", action="store_true", default=True, help="Whether to save the model weights in fp16")
    parser.add_argument(
        "--push_to_hub", action="store_true", default=False, help="Whether to push to HF Hub after saving"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    transformer = None
    vae = None

    if args.transformer_ckpt_path is not None:
        transformer = convert_transformer(args.transformer_ckpt_path)
    if args.vae_ckpt_path is not None:
        vae = convert_vae(args.vae_ckpt_path)

    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder_path, model_max_length=TOKENIZER_MAX_LENGTH)
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_path, cache_dir=args.text_encoder_cache_dir)

    scheduler = CogVideoXDDIMScheduler.from_config(
        {
            "snr_shift_scale": 3.0,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "num_train_timesteps": 1000,
            "prediction_type": "v_prediction",
            "rescale_betas_zero_snr": True,
            "set_alpha_to_one": True,
            "timestep_spacing": "linspace",
        }
    )

    pipe = CogVideoXPipeline(
        tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
    )

    if args.fp16:
        pipe = pipe.to(dtype=torch.float16)

    pipe.save_pretrained(args.output_path, safe_serialization=True, push_to_hub=args.push_to_hub)
