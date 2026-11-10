
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Locate the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from net.net import net

# Select device
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA-compatible NVIDIA GPU not detected. "
        "Please use a CUDA-enabled environment or Google Colab with GPU runtime."
    )

DEVICE = torch.device("cuda")

# Load pretrained model
MODEL_PATH = PROJECT_ROOT / "weights" / "PairLIE.pth"

model = net().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

model.load_state_dict(checkpoint, strict=True)
model.eval()


def enhance_image(image: Image.Image) -> Image.Image:
    """Enhance a low-light image using the pretrained PairLIE model."""

    image = image.convert("RGB")

    image_array = np.asarray(image, dtype=np.float32) / 255.0

    image_tensor = (
        torch.from_numpy(image_array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.inference_mode():
        L, R, X = model(image_tensor)
        enhanced = torch.pow(L, 0.2) * R

    enhanced = enhanced.squeeze(0).cpu().clamp(0, 1)

    output_array = (
        enhanced.permute(1, 2, 0).numpy() * 255
    ).astype(np.uint8)

    return Image.fromarray(output_array)
