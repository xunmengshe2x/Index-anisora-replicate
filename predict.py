import os
import sys

# Store the original path
original_path = sys.path.copy()

# Add the project root directory to sys.path before any imports
# This ensures both absolute imports (videosys.utils.logging) 
# and relative imports (.fastercache_sample_cogvideox_sp) work correctly
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now do your imports
import torch
import requests
from PIL import Image
from io import BytesIO
from huggingface_hub import hf_hub_download, snapshot_download
from anisoraV1_infer import CVModel

# Reset the path back to original if needed
# sys.path = original_path  # Commented out as we need to keep the path for submodule imports

REPO_ID = "IndexTeam/Index-anisora"
T5_VAE_DIR = "CogVideoX_VAE_T5"
MODEL_5B_DIR = "5B"

class Predictor:
    def setup(self):
        """Load the model into memory and ensure all required weights are present"""
        # Create necessary directories if they don't exist
        os.makedirs("pretrained_models", exist_ok=True)
        os.makedirs("ckpt", exist_ok=True)

        # Download T5 encoder and VAE weights
        print("Checking/downloading T5 encoder and VAE weights...")
        self._download_t5_vae()

        # Download 5B model weights
        print("Checking/downloading 5B model weights...")
        self._download_5b_weights()

        # Initialize the model
        print("Initializing model...")
        self.model = CVModel(n_gpus=1)

    def _download_t5_vae(self):
        """Download T5 encoder and VAE weights if not present"""
        try:
            # Download the entire T5 and VAE directory
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                local_dir="pretrained_models",
                allow_patterns=f"{T5_VAE_DIR}/*",
                ignore_patterns=["*.md", "*.txt"]
            )
            
            # Verify critical files exist
            required_files = [
                "t5-v1_1-xxl_new",
                "videokl_ch16_long_20w.pt"
            ]
            
            for file in required_files:
                full_path = os.path.join("pretrained_models", file)
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"Required file {file} not found after download")
                
        except Exception as e:
            raise RuntimeError(f"Failed to download T5/VAE weights: {str(e)}")

    def _download_5b_weights(self):
        """Download 5B model weights if not present"""
        try:
            # Download the 5B weights
            hf_hub_download(
                repo_id=REPO_ID,
                filename=f"{MODEL_5B_DIR}/1000/mp_rank_00_model_states.pt",
                local_dir="ckpt",
                repo_type="model"
            )
            
            # Verify the file exists
            if not os.path.exists("ckpt/mp_rank_00_model_states.pt"):
                raise FileNotFoundError("5B model weights not found after download")
                
        except Exception as e:
            raise RuntimeError(f"Failed to download 5B model weights: {str(e)}")

    def _process_image(self, image_path: str, target_size: int = 720) -> str:
        """Process input image - either URL or local path"""
        try:
            # Handle URL inputs
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_path)

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Center crop and resize
            #image = center_crop_arr(image, target_size)
            
            # Save processed image to temporary file
            temp_path = f"/tmp/{os.path.basename(image_path)}"
            image.save(temp_path)
            
            return temp_path

        except Exception as e:
            raise RuntimeError(f"Failed to process image {image_path}: {str(e)}")

    def predict(
        self,
        prompt: str,
        input_image: str,  # First frame
        input_image_mid: str = None,  # Optional middle frame
        input_image_last: str = None,  # Optional last frame
        motion: float = 0.7,
        gen_len: str = "3",
        seed: int = 554,
    ) -> str:
        """Run a single prediction on the model"""
        
        try:
            # Process input images
            resource = [self._process_image(input_image)]
            
            if input_image_mid:
                resource.append(self._process_image(input_image_mid))
            if input_image_last:
                resource.append(self._process_image(input_image_last))
            
            # If only one image provided, use it for all frames
            while len(resource) < 3:
                resource.append(resource[0])

            # Prepare output path
            output_path = "/tmp/output.mp4"
            
            # Set up parameters
            raw_params = {
                "Motion": motion,
                "gen_len": gen_len,
                "prompt": prompt,
                "seed": seed,
                "output_path": output_path
            }

            # Run inference
            result = self.model.run(resource, **raw_params)
            
            # Verify output exists
            if not os.path.exists(output_path):
                raise RuntimeError("Output video was not generated")
                
            return output_path

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def cleanup(self):
        """Clean up temporary files"""
        try:
            for file in os.listdir("/tmp"):
                if file.endswith((".jpg", ".png", ".mp4")):
                    os.remove(os.path.join("/tmp", file))
        except Exception as e:
            print(f"Warning: Failed to cleanup temporary files: {str(e)}")
