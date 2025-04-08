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
rgb_palette = np.array([[0, 0, 0],
                        [255, 0, 0],
                        [0, 255, 0],
                        [0, 0, 255],
                        [255, 255, 0]])

def rgb_to_5_channels(input_tensor):
    # Ensure input tensor is on the GPU if available
    device = input_tensor.device

    # Initialize 5 output channels as zeros
    output_tensor = torch.zeros((input_tensor.shape[0], 5, input_tensor.shape[2], input_tensor.shape[3]), device=device)

    # Get the RGB channels
    red = input_tensor[:, 0, :, :]   # Shape: (batch_size, height, width)
    green = input_tensor[:, 1, :, :] # Shape: (batch_size, height, width)
    blue = input_tensor[:, 2, :, :]  # Shape: (batch_size, height, width)

    # 1st channel: Red pixels (255, 0, 0)
    output_tensor[:, 0, :, :] = (red == 255) & (green == 0) & (blue == 0)
    # 2nd channel: Green pixels (0, 255, 0)
    output_tensor[:, 1, :, :] = (red == 0) & (green == 255) & (blue == 0)
    # 3rd channel: Blue pixels (0, 0, 255)
    output_tensor[:, 2, :, :] = (red == 0) & (green == 0) & (blue == 255)
    # 4th channel: Yellow pixels (255, 255, 0)
    output_tensor[:, 3, :, :] = (red == 255) & (green == 255) & (blue == 0)
    # 5th channel: Cyan pixels (0, 255, 255)
    output_tensor[:, 4, :, :] = (red == 0) & (green == 255) & (blue == 255)

    return output_tensor

