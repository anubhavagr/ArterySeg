import torch
import numpy as np

# Headings: Create Mask on GPU
def create_mask_gpu(image_tensor, device):
    assert image_tensor.shape == (4, 1344, 1344, 3), f"Expected shape (4, 1344, 1344, 3), but got {image_tensor.shape}"
    color_to_label = {
        (255, 0, 0): 1,
        (0, 255, 0): 2,
        (0, 0, 255): 3,
        (255, 255, 0): 4
    }
    image_flat = image_tensor.view(-1, 3)
    labels = torch.zeros(image_flat.shape[0], dtype=torch.int32, device=device)
    for color, label in color_to_label.items():
        color_tensor = torch.tensor(color, dtype=torch.int32, device=device)
        labels[torch.all(image_flat == color_tensor, dim=1)] = label
    mask = labels.view(image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2])
    return mask

# Headings: RGB Palette
import numpy as np
rgb_palette = np.array([[0, 0, 0],
                        [255, 0, 0],
                        [0, 255, 0],
                        [0, 0, 255],
                        [255, 255, 0]])

