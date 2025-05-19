from cog import BasePredictor, Input, Path, BaseModel
import torch
import os
import sys
from huggingface_hub import hf_hub_download
from PIL import Image
import shutil
from omegaconf import OmegaConf

# Add the anisoraV1_infer directory to the path
sys.path.append("anisoraV1_infer")

from cogvideox.pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import AutoencoderKLCogVideoX

class Output(BaseModel):
    video: Path

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        
        # Create necessary directories
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        try:
            # Download model components from HuggingFace
            model_id = "THUDM/CogVideoX-5b-I2V"
            
            # Download VAE
            vae = AutoencoderKLCogVideoX.from_pretrained(
                model_id,
                subfolder="vae",
                torch_dtype=torch.bfloat16
            )
            
            # Download text encoder and tokenizer
            tokenizer = T5Tokenizer.from_pretrained(
                model_id,
                subfolder="tokenizer"
            )
            
            text_encoder = T5EncoderModel.from_pretrained(
                model_id,
                subfolder="text_encoder",
                torch_dtype=torch.bfloat16
            )

            # Download transformer
            self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                model_id,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16
            )

            # Load config
            self.config = OmegaConf.load("anisoraV1_infer/configs/cogvideox/cogvideox_5b_720_169_2.yaml")

            # Move pipeline to GPU
            self.pipe.to("cuda")

        except Exception as e:
            raise RuntimeError(f"Failed to download model files: {str(e)}")

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

        try:
            # Set random seed
            torch.manual_seed(seed)
            
            # Load and preprocess image
            init_image = Image.open(str(image)).convert("RGB")
            
            # Generate video
            video_frames = self.pipe(
                prompt=prompt,
                image=init_image,
                height=720,
                width=1280,
                num_inference_steps=50,
                num_frames=13,
                guidance_scale=6.0,
                motion_scale=motion_scale
            ).frames[0]

            # Save video
            output_path = "results/output.mp4"
            video_frames.save(output_path, fps=8)

            if not os.path.exists(output_path):
                raise RuntimeError("Video generation failed - output file not found")

            return Output(video=Path(output_path))

        except Exception as e:
            raise RuntimeError(f"Video generation failed: {str(e)}")
