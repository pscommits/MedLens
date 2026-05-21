"""
vision.py
---------
Loads a torchxrayvision DenseNet model and runs inference on a chest X-ray.
Returns raw pathology probability scores.
"""

import skimage.io
import torch
import torchvision
import torchxrayvision as xrv


def load_model(weights="densenet121-res224-all"):
    """Load and return a pretrained torchxrayvision DenseNet model in eval mode."""
    model = xrv.models.DenseNet(weights=weights)
    model.eval()
    return model


def preprocess_image(img_path):
    """
    Load and preprocess a chest X-ray image.
    Returns a (1, 1, 224, 224) float tensor.
    """
    img = skimage.io.imread(img_path)

    # Normalize to xrv expected range [-1024, 1024]
    img = xrv.datasets.normalize(img, 255)

    # Grayscale: keep only first channel if RGB/RGBA
    if len(img.shape) > 2:
        img = img[:, :, 0]

    # Add channel dim -> (1, H, W)
    img = img[None, :, :]

    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])

    img = transform(img)

    # (1, 1, 224, 224) float tensor
    tensor = torch.from_numpy(img).unsqueeze(0).float()
    return tensor


def run_inference(model, img_tensor, threshold=None):
    """
    Run forward pass and return pathology scores as a sorted dict.

    Returns:
        Dict mapping pathology name -> probability score, sorted descending.
    """
    with torch.no_grad():
        preds = model(img_tensor)[0]

    scores = {
        pathology: float(pred)
        for pathology, pred in zip(model.pathologies, preds.detach().numpy())
    }

    # Sort by confidence descending
    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    if threshold is not None:
        scores = {k: v for k, v in scores.items() if v >= threshold}

    return scores