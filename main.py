import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from model import ArterySegModel, device
from dataset import StentDataset
from train import train, validate
from config import BATCH_SIZE, LR, EPOCHS, NUM_WORKERS, INPUT_PATH, VAL_INPUT_PATH, MASK_PATH, VAL_MASK_PATH, MODEL_NAME
from cbDice.loss import cldice_loss

torch.backends.cudnn.benchmark = True

model = ArterySegModel(in_ch=1, sobel_ch=64)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=6)
loss_fn = cldice_loss.SoftclDiceLoss()

def criterian(outputs, targets):
    return loss_fn(outputs, targets.long(), t_skeletonize_flage=True)

train_dataset = StentDataset(INPUT_PATH, MASK_PATH)
val_dataset = StentDataset(VAL_INPUT_PATH, VAL_MASK_PATH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                          drop_last=True, prefetch_factor=3, persistent_workers=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                        drop_last=True, prefetch_factor=3, persistent_workers=True, pin_memory=True)

train_losses = []
scaler = torch.cuda.amp.GradScaler()

for epoch in range(EPOCHS):
    train(model, train_loader, criterian, optimizer, scheduler, device, epoch, EPOCHS, train_losses, MODEL_NAME, scaler)
    validate(model, val_loader, device, MODEL_NAME, epoch)

