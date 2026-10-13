"""Opt-in adapters; no automatic downloads and no invented forensic weights."""
from pathlib import Path

import numpy as np


class DiffusionInpaintingAdapter:
    def __init__(self, local_model_directory, device='cpu'):
        if not Path(local_model_directory).is_dir() or not (Path(local_model_directory) / 'model_index.json').exists():
            raise FileNotFoundError('Provide an existing local Diffusers inpainting checkpoint; no weights are downloaded automatically')
        try:
            from diffusers import AutoPipelineForInpainting
        except ImportError as error:
            raise RuntimeError('Install the optional diffusion extra in a separate environment') from error
        self.pipeline = AutoPipelineForInpainting.from_pretrained(local_model_directory, local_files_only=True).to(device)
        self.device = device

    def edit(self, rgb, mask, prompt, seed):
        from PIL import Image
        import torch
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipeline(prompt=prompt, image=Image.fromarray(rgb), mask_image=Image.fromarray(mask.astype(np.uint8) * 255),
                               generator=generator).images[0]
        result = np.asarray(result.convert('RGB').resize((rgb.shape[1], rgb.shape[0]))).copy()
        result[mask == 0] = rgb[mask == 0]
        return result, mask.copy()


def load_trufor_prediction(path, expected_shape):
    """Import official inference output; does not substitute or implement TruFor itself."""
    with np.load(path, allow_pickle=False) as data:
        if 'map' not in data:
            raise ValueError('Expected official TruFor output NPZ containing map')
        prediction = np.asarray(data['map']).squeeze()
        if prediction.shape != expected_shape:
            raise ValueError('TruFor prediction does not match input raster/export dimensions')
        if not np.isfinite(prediction).all() or prediction.min() < 0 or prediction.max() > 1:
            raise ValueError('TruFor map must be a finite localization map in [0,1]')
        return prediction
