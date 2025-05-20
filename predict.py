import os
import torch
import sys 
import requests
from PIL import Image
from io import BytesIO
from huggingface_hub import hf_hub_download, snapshot_download

# Add the repository root to Python path BEFORE any imports
# This ensures all modules can be found during dynamic imports
sys.path.insert(0, '/src')

# Store the original path
original_path = sys.path.copy()

# Add the directory containing anisoraV1_infer to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Add the --base argument to sys.argv if it's not already there
if '--base' not in sys.argv:
    sys.argv.extend(['--base', '/src/anisoraV1_infer/configs/cogvideox/cogvideox_5b_720_169_2.yaml'])

# Print sys.path for debugging
print("DEBUG: sys.path at startup:", sys.path)

# Now do your imports
from anisoraV1_infer import CVModel

# Reset the path back to original
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
        os.makedirs(os.path.join("ckpt", "1000"), exist_ok=True)  # Create 1000 subdirectory

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
                local_dir="/src/pretrained_models",
                allow_patterns=f"{T5_VAE_DIR}/*",
                ignore_patterns=["*.md", "*.txt"]
            )
            
            # Verify critical files exist - FIXED: Check in the correct subdirectory
            required_files = [
                os.path.join(T5_VAE_DIR, "t5-v1_1-xxl_new"),
                os.path.join(T5_VAE_DIR, "videokl_ch16_long_20w.pt")
            ]
            
            for file in required_files:
                full_path = os.path.join("/src/pretrained_models", file)
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"Required file {file} not found after download")
                
            # Create symlinks to the expected locations if needed
            # This ensures compatibility with the model code that might expect files in specific locations
            if not os.path.exists("/src/pretrained_models/t5-v1_1-xxl_new"):
                os.symlink(
                    os.path.join("/src/pretrained_models", T5_VAE_DIR, "t5-v1_1-xxl_new"),
                    "pretrained_models/t5-v1_1-xxl_new"
                )
                
            if not os.path.exists("/src/pretrained_models/videokl_ch16_long_20w.pt"):
                os.symlink(
                    os.path.join("/src/pretrained_models", T5_VAE_DIR, "videokl_ch16_long_20w.pt"),
                    "/src/pretrained_models/videokl_ch16_long_20w.pt"
                )
                
        except Exception as e:
            raise RuntimeError(f"Failed to download T5/VAE weights: {str(e)}")

    def _download_5b_weights(self):
        """Download 5B model weights if not present"""
        try:
            # Download the 5B weights
            hf_hub_download(
                repo_id=REPO_ID,
                filename=f"{MODEL_5B_DIR}/1000/mp_rank_00_model_states.pt",
                local_dir="/src/ckpt",
                repo_type="model"
            )
            
            # Verify the file exists in the correct subdirectory
            expected_path = os.path.join("/src/ckpt", MODEL_5B_DIR, "1000", "mp_rank_00_model_states.pt")
            if os.path.exists(expected_path):
                # File downloaded to the correct location with subdirectories preserved
                # Create a symlink to the expected location if needed
                if not os.path.exists(os.path.join("/src/ckpt", "mp_rank_00_model_states.pt")):
                    os.symlink(
                        expected_path,
                        os.path.join("/src/ckpt", "mp_rank_00_model_states.pt")
                    )
            else:
                # Check if file was downloaded to a flattened path
                flattened_path = os.path.join("/src/ckpt", "mp_rank_00_model_states.pt")
                if os.path.exists(flattened_path):
                    # File exists at flattened path, create the directory structure and move the file
                    os.makedirs(os.path.dirname(expected_path), exist_ok=True)
                    os.rename(flattened_path, expected_path)
                    # Create a symlink back to the original expected location
                    os.symlink(expected_path, flattened_path)
                else:
                    # Check if file was downloaded to 1000 subdirectory directly
                    alt_path = os.path.join("/src/ckpt", "1000", "mp_rank_00_model_states.pt")
                    if os.path.exists(alt_path):
                        # Create a symlink to the expected location
                        if not os.path.exists(os.path.join("ckpt", "mp_rank_00_model_states.pt")):
                            os.symlink(
                                alt_path,
                                os.path.join("/src/ckpt", "mp_rank_00_model_states.pt")
                            )
                    else:
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
