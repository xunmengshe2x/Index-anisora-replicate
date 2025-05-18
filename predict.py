from cog import BasePredictor, Input, Path, BaseModel
import torch
import os
import wget
from huggingface_hub import hf_hub_download
from PIL import Image
import sys
import shutil

# Add the anisoraV1_infer directory to the path
sys.path.append("anisoraV1_infer")

from __init__ import CVModel

class Output(BaseModel):
    video: Path

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        # Create necessary directories
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        # Download model weights from HuggingFace
        # Based on repository links:
        # 🤗 Hugging Face: https://huggingface.co/IndexTeam/Index-anisora

        try:
            # Download VAE model
            vae_path = hf_hub_download(
                repo_id="PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
                filename="diffusion_pytorch_model.safetensors",
                subfolder="vae"
            )

            # Download other required model files
            # Note: Actual file names and paths would need to be extracted from the repository
            model_files = [
                "model.safetensors",
                "config.json",
                "tokenizer_config.json"
            ]

            for file in model_files:
                if not os.path.exists(f"checkpoints/{file}"):
                    hf_hub_download(
                        repo_id="IndexTeam/Index-anisora",
                        filename=file,
                        local_dir="checkpoints"
                    )

        except Exception as e:
            raise RuntimeError(f"Failed to download model files: {str(e)}")

        # Initialize the model
        try:
            self.model = CVModel(n_gpus=1)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def predict(
        self,
        image: Path = Input(description="Input image for video generation"),
        prompt: str = Input(description="Text prompt describing the desired video"),
        motion_scale: float = Input(
            description="Motion scale factor (0.7-1.3 recommended)",
            default=0.7,
            ge=0.1,
            le=2.0
        ),
        seed: int = Input(
            description="Random seed for generation",
            default=42
        )
    ) -> Output:
        """Run a single prediction on the model"""

        # Prepare parameters
        resource = [str(image)]
        raw_params = {
            "Motion": motion_scale,
            "gen_len": "3",
            "prompt": prompt,
            "seed": seed,
            "output_path": "results/output.mp4"
        }

        try:
            # Generate video
            result = self.model.run(resource, **raw_params)

            if not os.path.exists(raw_params["output_path"]):
                raise RuntimeError("Video generation failed - output file not found")

            return Output(video=Path(raw_params["output_path"]))

        except Exception as e:
            raise RuntimeError(f"Video generation failed: {str(e)}")
