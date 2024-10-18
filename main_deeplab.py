import warnings
warnings.filterwarnings('ignore')

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
from PIL import Image 
import os 
import glob as gb 
from tqdm.auto import tqdm

import torch 
import torch.nn as nn 
from torch.optim import Adam, SGD
import torchvision 
from torchvision import datasets 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchsummary import summary
from torchviz import make_dot
from models.shufflenet import MobileNetV2Segmentation
from dataset_utils import CustomData, walk_through_data, get_file_extensions, create_transform, create_dynamic_transform
from options import parse_train_args, parse_test_args
from models import network
from loss import FocalLoss, MultiClassDiceLoss, SoftDistillationLoss, KLDivergenceLoss
import torch.optim.lr_scheduler as lr_scheduler
from models.Model.model_import import model_import
args = parse_train_args()


# Dataset directories
train_images_dir = os.path.join(args.dataset_dir, 'train/train_image')
train_masks_dir = os.path.join(args.dataset_dir, 'train/train_mask')
val_images_dir = os.path.join(args.dataset_dir, 'val/val_image')
val_masks_dir = os.path.join(args.dataset_dir, 'val/val_mask')

# Validate image extensions
image_extensions = get_file_extensions(train_images_dir)
print(len(image_extensions), np.unique(image_extensions))

mask_extensions = get_file_extensions(train_masks_dir)
print(len(mask_extensions), np.unique(mask_extensions))

# Image size and transformation pipeline
image_size = args.image_size
data_transform = create_transform(image_size)

# Creating dataset instances
train_dataset = CustomData(train_images_dir, train_masks_dir, data_transform)
val_dataset = CustomData(val_images_dir, val_masks_dir, data_transform)

print(f"Number of training images: {len(train_dataset)}")
print(f"Number of validation images: {len(val_dataset)}")

# Dataloaders
train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=args.batch_size, 
    shuffle=True
)

val_loader = DataLoader(
    dataset=val_dataset, 
    batch_size=args.batch_size, 
    shuffle=False
)

# Fetch a sample batch
train_images, train_masks = next(iter(train_loader))
val_images, val_masks = next(iter(val_loader))

# Set device
torch.cuda.set_device(args.gpu_id)
torch.cuda.empty_cache()

device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Function to calculate model parameters
def calculate_param_count(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of parameters: {total_params}')

# Initialize the model
model = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=19, output_stride=16,pretrained_backbone=False).to(device)
calculate_param_count(model)
# model.load_state_dict(torch.load('Deeplab_checkpoint_epoch_400.pt', map_location=device))
kl_div_fn = KLDivergenceLoss()
focal_ls = FocalLoss()
dice_loss_fn = MultiClassDiceLoss()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=args.learning_rate * 0.0015, weight_decay=1e-5)
# optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)

scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

def validate_model(model, val_loader, criterion):
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for images, masks in tqdm(val_loader): 
            images, masks = images.to(device), masks.to(device)
            predictions = model(images)
            loss = criterion(predictions, masks) 
            validation_loss += loss.item()
    
    return validation_loss / len(val_loader)


def train_model(model, epochs, train_loader, optimizer, criterion, scheduler):
    training_loss_history = []

    for epoch in tqdm(range(epochs)): 
        model.train()
        epoch_loss = 0 
        data_transform = create_dynamic_transform(epoch, args.image_size)
        train_dataset = CustomData(train_images_dir, train_masks_dir, data_transform)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        
        for batch_idx, (images, masks) in enumerate(train_loader): 
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            predictions = model(images) 
            loss = 0.1 * focal_ls(predictions, masks) + 0.9 * criterion(predictions, masks) 
            
            if epoch > 20:
                loss = loss * 0.4 + 0.6 * dice_loss_fn(predictions, masks)
            
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx * len(images)}/{len(train_loader.dataset)} samples. loss: {loss.item()}")
                
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_avg_loss = epoch_loss / len(train_loader)
        training_loss_history.append(epoch_avg_loss)
        if (epoch) % 5 == 0:
            val_loss = validate_model(model, val_loader, criterion)
            print(f"Validation loss: {val_loss}")
            torch.save(model.state_dict(), f'Deeplab_checkpoint_epoch_{epoch}.pt')
        print(f"Epoch {epoch}: Loss = {epoch_avg_loss}\n")
        scheduler.step()
    
    return training_loss_history

training_loss = train_model(model, args.epochs, train_loader, optimizer, criterion, scheduler)
plt.plot(range(args.epochs), training_loss, color="blue", label="Training Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("training_loss_plot.png")
print(training_loss)
val_loss = validate_model(model, val_loader, criterion)
print(f"Validation loss: {val_loss}")
