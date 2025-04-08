import torch
import torch.nn.functional as F
import torch.nn as nn
from model_unet import device

# Laplacian Kernel for image processing
laplacian_kernel = torch.tensor([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

#laplacian_kernel = laplacian_kernel.  # Move to GPU if available

import torch
import torch.nn.functional as F

def laplacian_loss(pred, target):
    """
    Compute the Laplacian loss for each channel separately.

    Args:
        pred (torch.Tensor): Predicted tensor of shape (batch_size, channels, height, width).
        target (torch.Tensor): Target tensor of shape (batch_size, channels, height, width).

    Returns:
        torch.Tensor: Laplacian loss between the predicted and target tensors.
    """
    # Define the Laplacian kernel (3x3 kernel, weights)
    laplacian_kernel = torch.tensor([[0., 1., 0.],
                                     [1., -4., 1.],
                                     [0., 1., 0.]]).float().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 3, 3)
    
    # Move kernel to GPU if needed
    laplacian_kernel = laplacian_kernel.to(pred.device)

    # Initialize the Laplacian loss
    loss = 0.0

    # Apply the Laplacian kernel to each channel separately
    for i in range(pred.shape[1]):  # Iterate over the channels
        pred_channel = pred[:, i:i+1, :, :]  # Shape: (batch_size, 1, height, width)
        target_channel = target[:, i:i+1, :, :]  # Shape: (batch_size, 1, height, width)

        # Compute the difference for the current channel
        pred_diff = pred_channel - target_channel

        # Apply the Laplacian kernel to the predicted difference
        pred_diff_laplacian = F.conv2d(pred_diff, laplacian_kernel, padding=1)

        # Accumulate the loss (mean squared error of Laplacian)
        loss += torch.mean(pred_diff_laplacian ** 2)

    return loss

class HybridLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        # Binary Cross-Entropy Loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target)
        
        # Laplacian Loss
        lap_loss = laplacian_loss(pred, target)
        
        # Combine the losses
        total_loss = self.alpha * bce_loss + self.beta * lap_loss
        return total_loss
