"""
gcam.py
-------
GradCAM implementation for torchxrayvision DenseNet models.

Target layer: model.features.denseblock4
  - Last convolutional block before global average pooling
  - Best layer for spatial heatmaps on DenseNet121
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class _GradCamHooks:
    """Registers and stores forward activations + backward gradients."""

    def __init__(self):
        self.activations = None
        self.gradients = None
        self._handles = []

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


class GradCAM:
    """
    GradCAM for torchxrayvision DenseNet.

    Usage:
        gcam = GradCAM(model)
        heatmap = gcam.compute(img_tensor, "Cardiomegaly")
    """

    def __init__(self, model, target_layer="features.denseblock4"):
        self.model = model

        # Walk the dot-path to get the actual layer
        layer = model
        for part in target_layer.split("."):
            layer = getattr(layer, part)
        self.target_layer = layer

        self._hooks = _GradCamHooks()

    def compute(self, img_tensor, pathology):
        """
        Compute a GradCAM heatmap for a single pathology.

        Args:
            img_tensor: Preprocessed tensor (1, 1, 224, 224)
            pathology:  Pathology name string, must exist in model.pathologies

        Returns:
            Float32 numpy array (224, 224) in range [0, 1]
        """
        if pathology not in self.model.pathologies:
            raise ValueError(
                f"'{pathology}' not in model.pathologies. "
                f"Available: {list(self.model.pathologies)}"
            )

        class_idx = list(self.model.pathologies).index(pathology)

        self._hooks.register(self.target_layer)
        try:
            heatmap = self._run(img_tensor, class_idx)
        finally:
            self._hooks.remove()

        return heatmap

    def _run(self, img_tensor, class_idx):
        self.model.zero_grad()
        logits = self.model(img_tensor)
        logits[0, class_idx].backward()

        activations = self._hooks.activations  # (1, C, h, w)
        gradients = self._hooks.gradients      # (1, C, h, w)

        if activations is None or gradients is None:
            raise RuntimeError("GradCAM hooks captured no data.")

        # Global average pool gradients -> weights per channel
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activation maps + ReLU
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))

        # Upsample to 224x224
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        lo, hi = cam.min(), cam.max()
        cam = (cam - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(cam)

        return cam.astype(np.float32)


def save_heatmap(original_img, heatmap, output_path, pathology, score, alpha=0.4):
    """
    Overlay GradCAM heatmap on the original image and save to disk.

    Args:
        original_img: uint8 image array from skimage (grayscale or RGB)
        heatmap:      float32 (224, 224) array in [0, 1]
        output_path:  path to write the PNG
        pathology:    label text to draw on image
        score:        probability score to draw on image
        alpha:        heatmap blend opacity
    """
    # Convert to BGR
    if len(original_img.shape) == 2:
        img_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

    img_bgr = cv2.resize(img_bgr, (224, 224))

    # Colorise and blend
    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, colored, alpha, 0)

    # Draw label
    label = f"{pathology}: {score:.3f}"
    cv2.putText(overlay, label, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(overlay, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(output_path, overlay)
    print(f"  Saved: {output_path}")