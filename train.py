import os
import time
import numpy as np
import cv2
import torch
from tqdm import tqdm
from utils import create_mask_gpu, rgb_palette

# Headings: Training Function
def train(model, train_loader, criterion, optimizer, scheduler, device, epoch, EPOCHS, train_losses, model_name, scaler):
    model.train()
    if epoch > 30:
        model.backbone.requires_grad_(True)
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS}")
    for images, masks in progress_bar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        one_hot_mask = create_mask_gpu(masks, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, one_hot_mask.long())
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
        progress_bar.set_postfix(loss=running_loss / len(train_loader.dataset))
    scheduler.step(running_loss)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.5f}")
    if epoch % 10 == 0:
        os.makedirs("logs", exist_ok=True)
        torch.save(model.state_dict(), f"logs/{epoch}_finalized_seg_{model_name}.pth")

# Headings: Validation Function
def validate(model, val_loader, device, model_name, epoch):
    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            break
    images, outputs = images.cpu(), outputs.cpu()
    binary_outputs = torch.argmax(outputs, axis=1)
    binary_outputs = rgb_palette[binary_outputs]
    if np.sum(binary_outputs) == 0:
        print("Warning: All predicted outputs are zero.")
    batch_size = images.shape[0]
    output_images = []
    for idx in range(batch_size):
        original_image = images[idx][0]
        mask = masks[idx]
        predicted_mask = binary_outputs[idx]
        if len(original_image.shape) == 2:
            original_image = np.expand_dims(original_image, axis=-1)
            original_image = np.repeat(original_image, 3, axis=-1)
        elif len(original_image.shape) == 3 and original_image.shape[2] == 1:
            original_image = np.repeat(original_image, 3, axis=-1)
        if len(mask.shape) == 3 and mask.shape[2] == 3:
            pass
        elif len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
            mask = np.repeat(mask, 3, axis=-1)
        if len(predicted_mask.shape) == 3 and predicted_mask.shape[2] == 3:
            pass
        elif len(predicted_mask.shape) == 2:
            predicted_mask = np.expand_dims(predicted_mask, axis=-1)
            predicted_mask = np.repeat(predicted_mask, 3, axis=-1)
        original_image = np.uint8(original_image * 255)
        mask = np.uint8(mask)
        predicted_mask = np.uint8(predicted_mask)
        combined_image = np.concatenate([original_image, mask, predicted_mask], axis=1)
        output_images.append(combined_image)
    final_image = np.concatenate(output_images, axis=0)
    os.makedirs(model_name, exist_ok=True)
    cv2.imwrite(f"{model_name}/{model_name}_{epoch}.png", final_image)

