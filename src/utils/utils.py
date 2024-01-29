from typing import Optional
from PIL import Image
from numpy.typing import NDArray

import torch

from ..core.model_structure import UNet
from .setup_logger import logger


def check_cuda_availability() -> str:
    """
    Check if CUDA (GPU) available and set device respectively
    """
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA is available and set as device")

    else:
        device = "cpu"
        logger.info("CUDA is not available. CPU is set as device")

    return device


def load_model(path) -> Optional[UNet]:
    """
    Load trained UNet model
    """
    try:
        device = check_cuda_availability()

        model = UNet()
        model.load_state_dict(
            torch.load(
                path,
                map_location=torch.device(device),
            )
        )
        logger.info("The model has been loaded")

        return model

    except FileNotFoundError:
        logger.warning("Error: File not found")


def predict(model: UNet, image: NDArray) -> Optional[Image.Image]:
    """
    Image color prediction
    """

    img_normalized = image / 50.0 - 1
    img_tensor = torch.tensor(img_normalized).float().unsqueeze(0).unsqueeze(0)

    try:
        image_pred = model.predict(img_tensor)

        logger.info("The color image was generated")

        return image_pred

    except torch.TensorError as e:
        logger.error(f"Tensor error: {e}")
