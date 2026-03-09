"""
Anomalib Strategy Module for MicroSentryAI.

This module provides the concrete implementation of the anomaly detection strategy.
It primarily utilizes Anomalib's `TorchInferencer` for execution. It also
implements a robust fallback to standard PyTorch (`torch.load`) with a dynamic 
unpickler to bypass missing custom checkpoint classes if the provided 
model was not trained within the Anomalib ecosystem.
"""

import os
import io
import pickle
import logging
from pathlib import Path
from typing import Tuple, Optional

# Enable MPS Fallback for Apple Silicon users
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
from anomalib.deploy import TorchInferencer
import cv2

logger = logging.getLogger(__name__)


# =========================================================================
# DYNAMIC UNPICKLER (Bypasses missing classes in torch.load)
# =========================================================================

class DummyMeta(type):
    def __getattr__(cls, name):
        return DummyClass

class DummyClass(metaclass=DummyMeta):
    """A completely inert class that absorbs all PyTorch initialization calls."""
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return DummyClass()
    def __getattr__(self, name): return DummyClass()
    def __getitem__(self, key): return DummyClass()
    def __setitem__(self, key, value): pass
    def __setstate__(self, state): pass

class DynamicUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError, ModuleNotFoundError) as e:
            logger.warning(f"Dynamically mocking missing checkpoint class: {module}.{name}")
            return DummyClass

class DynamicPickleModule:
    """A drop-in replacement for the `pickle` module used by torch.load."""

    Unpickler = DynamicUnpickler

    @staticmethod
    def load(file, **kwargs):
        return DynamicUnpickler(file).load()

    @staticmethod
    def loads(b, **kwargs):
        return DynamicUnpickler(io.BytesIO(b)).load()

# =========================================================================


class AnomalibStrategy:
    """
    Strategy for executing PyTorch (.pt, .ckpt) anomaly detection models.
    """

    def __init__(self):
        self.torch_inferencer: Optional[TorchInferencer] = None
        self.raw_model = None  # Fallback handler for non-Anomalib models
        
        self.device = "auto"
        self.model_type = "unknown" 
        self.model_name = "Unknown"
        self._device_verified = False 

    def set_device(self, device_code: str):
        """Sets the target hardware device for inference."""
        self.device = device_code.lower()
        logger.info(f"Target Device set to: {self.device}")

    def _resolve_device(self) -> str:
        """Determines the optimal available hardware device if set to 'auto'."""
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_from_file(self, model_path: str):
        """
        Loads PyTorch models, attempting Anomalib's interface first, 
        with a fallback to standard torch.load using the Dynamic Unpickler.
        """
        path = Path(model_path)
        self._device_verified = False
        self.torch_inferencer = None
        self.raw_model = None
        
        try:
            os.environ["TRUST_REMOTE_CODE"] = "1" 

            if path.suffix not in [".pt", ".ckpt"]:
                raise ValueError(f"Unsupported file type: {path.suffix}. Expected .pt or .ckpt")
                
            self.model_type = "torch"
            resolved_device = self._resolve_device()
            final_device = resolved_device.upper()
            
           # --- ATTEMPT 1: Anomalib's TorchInferencer ---
            import functools
            original_torch_load = torch.load
            
            try:
                try:
                    # MONKEY PATCH: Force all tensors to CPU during deserialization
                    torch.load = functools.partial(original_torch_load, map_location="cpu")
                    
                    if resolved_device == "mps":
                        logger.debug("Applying MPS Shim: Initializing on CPU first.")
                        self.torch_inferencer = TorchInferencer(path=path, device="cpu")
                        
                        mps_device = torch.device("mps")
                        if hasattr(self.torch_inferencer, 'model'):
                            self.torch_inferencer.model = self.torch_inferencer.model.to(mps_device)
                        self.torch_inferencer.device = mps_device
                        final_device = "MPS (Apple Silicon)"
                    else:
                        self.torch_inferencer = TorchInferencer(path=path, device=resolved_device)
                    
                    self.model_name = f"Anomalib (Torch) [{final_device}]"
                    logger.info(f"Successfully loaded {self.model_name} via TorchInferencer")
                    
                finally:
                    # ALWAYS restore the original torch.load behavior so it doesn't break Attempt 2
                    torch.load = original_torch_load

            # --- ATTEMPT 2: Raw PyTorch Fallback with Dynamic Unpickler ---
            except Exception as anomalib_err:
                logger.warning(f"TorchInferencer rejected the model: {anomalib_err}. Falling back to Dynamic Unpickler.")
                self.torch_inferencer = None
                
                # Setup device safely
                device_obj = torch.device(resolved_device if resolved_device != "mps" else "cpu")
                
                # Use the custom pickle module to bypass missing classes
                loaded_data = torch.load(path, map_location=device_obj, pickle_module=DynamicPickleModule)
                
                # NEW FIX: Unpack Anomalib's dictionary wrapper
                if isinstance(loaded_data, dict):
                    if "state_dict" in loaded_data and "model" not in loaded_data:
                        raise ValueError(
                            "CRITICAL ERROR: You loaded the training .ckpt file! "
                            "Please select the exported model.pt file from your weights/torch folder."
                        )
                    elif "model" in loaded_data:
                        self.raw_model = loaded_data["model"]
                        logger.info("Successfully extracted model graph from dictionary wrapper.")
                    else:
                        raise ValueError("The loaded dictionary does not contain a recognizable model graph.")
                else:
                    self.raw_model = loaded_data

                if hasattr(self.raw_model, 'eval'):
                    self.raw_model.eval()
                    
                if resolved_device == "mps":
                    try:
                        self.raw_model = self.raw_model.to(torch.device("mps"))
                        final_device = "MPS (Apple Silicon)"
                    except Exception as mps_err:
                        logger.debug(f"Failed to push raw model to MPS: {mps_err}")
                
                self.model_name = f"Raw PyTorch Model [{final_device}]"
                logger.info(f"Successfully loaded {self.model_name} via standard torch.load")

        except Exception as e:
            logger.error(f"Critical failure loading model: {e}")
            raise RuntimeError(f"Load Error: {e}")

    def predict(self, image_path: str) -> Tuple[float, np.ndarray]:
        """Routes the inference request to the successful loading method."""
        if self.torch_inferencer:
            return self._predict_anomalib(image_path)
        elif self.raw_model:
            return self._predict_raw(image_path)
            
        return 0.0, np.zeros((256, 256), dtype=np.float32)

    def _predict_anomalib(self, image_path: str) -> Tuple[float, np.ndarray]:
        """Executes inference using Anomalib's highly-structured pipeline."""
        try:
            if not self._device_verified:
                try:
                    p = next(self.torch_inferencer.model.parameters())
                    logger.debug(f"Verification: Tensors are on {p.device}")
                    self._device_verified = True
                except StopIteration:
                    pass

            result = self.torch_inferencer.predict(image=image_path)

            if self.device == "mps" and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()

            score = 0.0
            if hasattr(result, 'pred_score') and result.pred_score is not None:
                score = float(result.pred_score.item() if isinstance(result.pred_score, torch.Tensor) else result.pred_score)

            heatmap = result.anomaly_map
            if isinstance(heatmap, torch.Tensor):
                heatmap = heatmap.detach().cpu().numpy()
            
            heatmap = heatmap.squeeze()
            
            if heatmap.max() > heatmap.min():
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

            return score, heatmap.astype(np.float32)

        except Exception as e:
            logger.error(f"Anomalib Inference failed: {e}")
            return 0.0, np.zeros((256, 256), dtype=np.float32)

    def _predict_raw(self, image_path: str) -> Tuple[float, np.ndarray]:
        """
        Executes a best-effort forward pass and parses Anomalib's raw tuple outputs.
        """
        try:
            # Standard Preprocessing
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            
            # Convert to Tensor (CHW format)
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            device_obj = next(self.raw_model.parameters()).device if hasattr(self.raw_model, 'parameters') else torch.device('cpu')
            tensor = tensor.to(device_obj)
            
            with torch.no_grad():
                output = self.raw_model(tensor)
            
            score = 0.0
            heatmap = np.zeros((256, 256), dtype=np.float32)

            # Parse Anomalib's standard raw output tuple: (anomaly_map, score)
            if isinstance(output, tuple):
                for item in output:
                    if isinstance(item, torch.Tensor):
                        # FIX 1: Only grab floating-point tensors for the heatmap to avoid boolean masks
                        if item.ndim >= 2 and item.numel() > 1 and item.is_floating_point():
                            heatmap = item.squeeze().cpu().numpy()
                        elif item.numel() == 1:  # Scalar Score
                            score = float(item.cpu().item())
            elif isinstance(output, torch.Tensor):
                heatmap = output.squeeze().cpu().numpy()
                
            # FIX 2: Explicitly cast to float32 before any math operations
            heatmap = heatmap.astype(np.float32)

            # Normalize to 0-1 range for the MicroSentry GUI overlay
            if heatmap.max() > heatmap.min():
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

            return score, heatmap
            
        except Exception as e:
            logger.error(f"Raw PyTorch Inference failed: {e}")
            return 0.0, np.zeros((256, 256), dtype=np.float32)