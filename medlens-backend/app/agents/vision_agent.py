"""
vision_agent.py
---------------
Integrates vision.py (TorchXRayVision inference) and gcam.py (GradCAM heatmap).

Called by app/main.py as:
    pathologies, heatmap_b64 = await run_vision_and_gradcam(image_bytes)

Input:
    image_bytes : raw bytes of the uploaded X-ray image (JPG or PNG)

Output:
    pathologies     : dict  — { "Atelectasis": 0.88, "Effusion": 0.14, ... }
                              sorted by confidence descending, all pathologies included
    heatmap_b64     : str   — data-URI string  "data:image/png;base64,<encoded PNG>"
                              GradCAM overlay of the TOP-1 pathology, ready for frontend display
"""

import io
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import skimage.io
import torch
import torchvision
import torchxrayvision as xrv

from gcam import GradCAM, save_heatmap   # your existing gcam.py
from vision import load_model, preprocess_image, run_inference  # your existing vision.py


# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------
# We load the heavy DenseNet model once when the FastAPI app starts up,
# then reuse the same object for every request. This avoids reloading
# ~100 MB of weights on every HTTP call.

_model = None
_executor = ThreadPoolExecutor(max_workers=2)   # runs blocking PyTorch code off the async loop


def _get_model():
    """Lazy-load and cache the TorchXRayVision DenseNet model."""
    global _model
    if _model is None:
        print("[vision_agent] Loading TorchXRayVision DenseNet121...")
        _model = load_model(weights="densenet121-res224-all")
        print("[vision_agent] Model loaded.")
    return _model


# ---------------------------------------------------------------------------
# Internal synchronous helper (runs in thread-pool)
# ---------------------------------------------------------------------------

def _run_sync(image_bytes: bytes):
    """
    Synchronous core: inference + GradCAM.
    Runs inside a ThreadPoolExecutor so it doesn't block FastAPI's event loop.

    Steps:
        1. Decode image bytes  →  numpy array
        2. Preprocess          →  (1,1,224,224) float tensor
        3. Run inference       →  pathology scores dict
        4. GradCAM on top-1    →  (224,224) float32 heatmap
        5. Overlay heatmap     →  PNG → base64 data-URI string

    Returns:
        (pathologies, heatmap_b64)
    """
    model = _get_model()

    # --- Step 1: Decode bytes to numpy image ---
    # skimage.io.imread expects a file path or file-like object
    img_array = skimage.io.imread(io.BytesIO(image_bytes))

    # --- Step 2: Preprocess for DenseNet ---
    # We replicate the logic in vision.py but starting from an array (not a path)
    img_norm = xrv.datasets.normalize(img_array, 255)

    if len(img_norm.shape) > 2:
        img_norm = img_norm[:, :, 0]          # keep first channel if RGB/RGBA

    img_norm = img_norm[None, :, :]            # add channel dim → (1, H, W)

    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    img_norm = transform(img_norm)
    img_tensor = torch.from_numpy(img_norm).unsqueeze(0).float()  # (1,1,224,224)

    # --- Step 3: Inference ---
    pathologies = run_inference(model, img_tensor, threshold=None)
    # pathologies is already sorted descending by score

    # --- Step 4: GradCAM on top pathology ---
    top_pathology = next(iter(pathologies))          # highest-confidence class
    top_score     = pathologies[top_pathology]

    gcam = GradCAM(model)
    try:
        heatmap = gcam.compute(img_tensor, top_pathology)   # (224,224) float32
    except Exception as e:
        print(f"[vision_agent] GradCAM failed for {top_pathology}: {e}")
        heatmap = np.zeros((224, 224), dtype=np.float32)    # graceful fallback

    # --- Step 5: Overlay and encode as base64 data-URI ---
    heatmap_b64 = _heatmap_to_data_uri(img_array, heatmap, top_pathology, top_score)

    return pathologies, heatmap_b64


def _heatmap_to_data_uri(original_img, heatmap, pathology, score, alpha=0.4):
    """
    Blend the GradCAM heatmap on the original image and return a PNG data-URI.

    Instead of writing to disk (save_heatmap does that), we encode in-memory
    so the string can go directly into the API response JSON.
    """
    # Ensure BGR for OpenCV
    if len(original_img.shape) == 2:
        img_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

    img_bgr = cv2.resize(img_bgr, (224, 224))

    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, colored, alpha, 0)

    label = f"{pathology}: {score:.3f}"
    cv2.putText(overlay, label, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(overlay, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Encode to PNG bytes in memory (no temp file needed)
    success, buffer = cv2.imencode(".png", overlay)
    if not success:
        raise RuntimeError("cv2.imencode failed — could not encode heatmap PNG")

    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Public async entry-point  (called by app/main.py)
# ---------------------------------------------------------------------------

async def run_vision_and_gradcam(image_bytes: bytes):
    """
    Async wrapper.  FastAPI's event loop stays free while PyTorch runs in a thread.

    Args:
        image_bytes : raw bytes read from the UploadFile

    Returns:
        pathologies  : dict[str, float]   e.g. {"Atelectasis": 0.88, ...}
        heatmap_b64  : str                "data:image/png;base64,..."
    """
    loop = asyncio.get_event_loop()
    pathologies, heatmap_b64 = await loop.run_in_executor(
        _executor, _run_sync, image_bytes
    )
    return pathologies, heatmap_b64