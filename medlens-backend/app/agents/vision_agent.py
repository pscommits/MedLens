"""
vision_agent.py
---------------
Self-contained vision agent for MedLens backend.
Integrates TorchXRayVision DenseNet inference + GradCAM heatmap generation.

No external .py imports — all logic is inlined here.

Called by app/main.py as:
    pathologies, heatmap_b64 = await run_vision_and_gradcam(image_bytes)

Input:
    image_bytes : bytes  — raw bytes of the uploaded X-ray image (JPG or PNG)

Output:
    pathologies : dict[str, float]  — e.g. {"Atelectasis": 0.88, "Effusion": 0.14, ...}
                                       all pathologies, sorted by confidence descending
    heatmap_b64 : str               — "data:image/png;base64,..."
                                       GradCAM overlay of the top-1 pathology,
                                       ready to drop into an HTML <img src="..."> tag
"""

from __future__ import annotations

import io
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchxrayvision as xrv
import skimage.io


# ---------------------------------------------------------------------------
# Thread pool — PyTorch inference is blocking CPU/GPU work.
# Running it inside async def without an executor would freeze FastAPI
# and block every other incoming request until inference finishes.
# The executor moves that work to a background thread instead.
# ---------------------------------------------------------------------------
_executor = ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------------------
# Model cache — DenseNet121 is ~100 MB and takes ~3 s to load.
# We load it once when the first request arrives, then reuse forever.
# ---------------------------------------------------------------------------
_model = None


def _get_model(weights: str = "densenet121-res224-all"):
    global _model
    if _model is None:
        print(f"[vision_agent] Loading TorchXRayVision DenseNet ({weights})...")
        _model = xrv.models.DenseNet(weights=weights)
        _model.eval()
        print("[vision_agent] Model ready.")
    return _model


# ===========================================================================
# SECTION 1 — GradCAM  (inlined from gcam.py)
# ===========================================================================

class _GradCamHooks:
    """Registers PyTorch hooks to capture forward activations and backward gradients."""

    def __init__(self):
        self.activations = None
        self.gradients   = None
        self._handles    = []

    def register(self, layer):
        self._handles.append(layer.register_forward_hook(self._save_activations))
        self._handles.append(layer.register_full_backward_hook(self._save_gradients))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()


class _GradCAM:
    """
    GradCAM for torchxrayvision DenseNet121.

    Target layer : model.features.denseblock4
        — last convolutional block before global average pooling
        — produces the best spatial heatmaps for DenseNet121
    """

    def __init__(self, model, target_layer: str = "features.denseblock4"):
        self.model = model
        # Walk the dot-path to reach the actual nn.Module layer object
        layer = model
        for part in target_layer.split("."):
            layer = getattr(layer, part)
        self.target_layer = layer
        self._hooks = _GradCamHooks()

    def compute(self, img_tensor: torch.Tensor, pathology: str) -> np.ndarray:
        """
        Compute a GradCAM heatmap for one pathology class.

        Args:
            img_tensor : (1, 1, 224, 224) float tensor — preprocessed X-ray
            pathology  : pathology name string, must exist in model.pathologies

        Returns:
            float32 numpy array (224, 224) with values in [0, 1]
        """
        if pathology not in self.model.pathologies:
            raise ValueError(
                f"'{pathology}' not found in model.pathologies. "
                f"Available: {list(self.model.pathologies)}"
            )

        class_idx = list(self.model.pathologies).index(pathology)
        self._hooks.register(self.target_layer)
        try:
            heatmap = self._run(img_tensor, class_idx)
        finally:
            self._hooks.remove()
        return heatmap

    def _run(self, img_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(img_tensor)
        logits[0, class_idx].backward()

        activations = self._hooks.activations   # (1, C, h, w)
        gradients   = self._hooks.gradients     # (1, C, h, w)

        if activations is None or gradients is None:
            raise RuntimeError("[vision_agent] GradCAM hooks captured no data.")

        # Global-average-pool the gradients to get per-channel weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps + ReLU (only positive influence)
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))

        # Upsample from feature-map size (~7x7) to full image size (224x224)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        lo, hi = cam.min(), cam.max()
        cam = (cam - lo) / (hi - lo) if (hi - lo) > 1e-8 else np.zeros_like(cam)
        return cam.astype(np.float32)


# ===========================================================================
# SECTION 2 — Image preprocessing  (inlined from vision.py)
# ===========================================================================

def _preprocess(image_bytes: bytes) -> tuple[np.ndarray, torch.Tensor]:
    """
    Decode raw image bytes and prepare them for DenseNet inference.

    Steps:
        1. Decode bytes → numpy array via skimage
        2. Normalize pixel values to xrv's expected range [-1024, 1024]
        3. Convert to grayscale (keep first channel if RGB/RGBA)
        4. Add channel dimension → (1, H, W)
        5. Apply XRayCenterCrop + XRayResizer(224)
        6. Wrap in a (1, 1, 224, 224) float tensor

    Returns:
        original_img : uint8 numpy array  — kept for heatmap overlay
        img_tensor   : (1, 1, 224, 224) float32 torch.Tensor
    """
    original_img = skimage.io.imread(io.BytesIO(image_bytes))

    img = xrv.datasets.normalize(original_img, 255)   # → float in [-1024, 1024]

    if len(img.shape) > 2:
        img = img[:, :, 0]          # drop colour channels, keep first

    img = img[None, :, :]           # add channel dim → (1, H, W)

    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    img = transform(img)

    img_tensor = torch.from_numpy(img).unsqueeze(0).float()   # (1, 1, 224, 224)
    return original_img, img_tensor


# ===========================================================================
# SECTION 3 — Inference  (inlined from vision.py)
# ===========================================================================

def _run_inference(model, img_tensor: torch.Tensor) -> dict[str, float]:
    """
    Forward pass through DenseNet.

    Returns:
        dict mapping pathology name → probability score, sorted descending.
    """
    with torch.no_grad():
        preds = model(img_tensor)[0]

    scores = {
        pathology: float(pred)
        for pathology, pred in zip(model.pathologies, preds.detach().numpy())
    }
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


# ===========================================================================
# SECTION 4 — Heatmap overlay + base64 encode  (inlined from gcam.py)
# ===========================================================================

def _overlay_to_data_uri(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    pathology: str,
    score: float,
    alpha: float = 0.4,
) -> str:
    """
    Blend the GradCAM heatmap on the original image and return a PNG data-URI.

    We encode in-memory (no temp file) so the result goes directly into
    the JSON response. The frontend renders it via:
        <img src="data:image/png;base64,..." />

    Args:
        original_img : uint8 numpy array (grayscale or RGB)
        heatmap      : float32 (224, 224) array in [0, 1]
        pathology    : label to draw on the overlay image
        score        : probability score to draw on the overlay image
        alpha        : heatmap opacity blend factor (0 = invisible, 1 = full)

    Returns:
        str  —  "data:image/png;base64,<encoded bytes>"
    """
    # Ensure BGR for OpenCV
    if len(original_img.shape) == 2:
        img_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

    img_bgr = cv2.resize(img_bgr, (224, 224))

    # Colourise heatmap with JET colormap and blend over original
    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, colored, alpha, 0)

    # Draw pathology label + score as text on the image
    label = f"{pathology}: {score:.3f}"
    cv2.putText(overlay, label, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0),   1, cv2.LINE_AA)
    cv2.putText(overlay, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Encode PNG in memory → base64 string
    success, buffer = cv2.imencode(".png", overlay)
    if not success:
        raise RuntimeError("[vision_agent] cv2.imencode failed — could not produce PNG.")

    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ===========================================================================
# SECTION 5 — Synchronous pipeline (runs in thread pool)
# ===========================================================================

def _run_sync(image_bytes: bytes) -> tuple[dict, str]:
    """
    Full synchronous pipeline: decode → preprocess → infer → GradCAM → encode.

    This function is blocking (PyTorch work) so it must NOT be called
    directly in an async context. It is dispatched via run_in_executor below.
    """
    model = _get_model()

    # 1. Preprocess
    original_img, img_tensor = _preprocess(image_bytes)

    # 2. Inference
    pathologies = _run_inference(model, img_tensor)

    # 3. GradCAM on top-1 pathology
    top_pathology = next(iter(pathologies))
    top_score     = pathologies[top_pathology]

    gcam = _GradCAM(model)
    try:
        heatmap = gcam.compute(img_tensor, top_pathology)
    except Exception as e:
        print(f"[vision_agent] GradCAM failed for '{top_pathology}': {e}. Using blank heatmap.")
        heatmap = np.zeros((224, 224), dtype=np.float32)

    # 4. Overlay + encode
    heatmap_b64 = _overlay_to_data_uri(original_img, heatmap, top_pathology, top_score)

    return pathologies, heatmap_b64


# ===========================================================================
# PUBLIC ENTRY POINT  —  called by app/main.py
# ===========================================================================

async def run_vision_and_gradcam(image_bytes: bytes) -> tuple[dict, str]:
    """
    Async entry point called by the FastAPI orchestrator in app/main.py.

    Dispatches the blocking PyTorch work to a thread pool so FastAPI's
    event loop stays free to handle other requests concurrently.

    Args:
        image_bytes : bytes — raw content from UploadFile.read()

    Returns:
        pathologies : dict[str, float]   {"Atelectasis": 0.88, "Effusion": 0.14, ...}
        heatmap_b64 : str                "data:image/png;base64,..."
    """
    loop = asyncio.get_event_loop()
    pathologies, heatmap_b64 = await loop.run_in_executor(
        _executor, _run_sync, image_bytes
    )
    return pathologies, heatmap_b64