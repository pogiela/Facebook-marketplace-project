from create_pytorch_dataset import CustomImageDataset 
import torch
import torchvision.models as models
import numpy
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import pickle
import os
import time
from tempfile import TemporaryDirectory
from torch.optim import lr_scheduler
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Variables
file_path = 'cleaned_data/training_data.csv'
img_dir = "cleaned_data/images/"

# Load decoder dictionary from file
with open("image_decoder.pkl", "rb") as f:
    decoder_dict = pickle.load(f)

# Extract the encoder and decoder from the loaded dictionary
encoder = decoder_dict['encoder']
decoder = decoder_dict['decoder']

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),  # Optional data augmentation
    transforms.RandomRotation(10),      # Optional data augmentation
])


# Create dataset
full_dataset = CustomImageDataset(annotations_file=file_path, img_dir=img_dir, transform=transform)

# Split dataset into training (70%), validation (15%), and test (15%) sets
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f'Training dataset size: {len(train_dataset)}')
print(f'Validation dataset size: {len(val_dataset)}')
print(f'Test dataset size: {len(test_dataset)}')


# Load the pre-trained ResNet-50 model with the 'weights' parameter
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Get the number of input features for the final linear layer
num_features = model.fc.in_features

# Replace the final linear layer with a new one (num_classes is the number of categories)
num_classes = len(decoder)
model.fc = torch.nn.Linear(num_features, num_classes)

# Load the best model weighths from training
# best_model_weights_path = 'best_model_params.pt'
best_model_weights_path = 'model_evaluation/model_20240803-193204/weights/epoch_24_weights.pth'
if torch.cuda.is_available():
    model.load_state_dict(torch.load(best_model_weights_path, weights_only=True))
else:
    model.load_state_dict(torch.load(best_model_weights_path, weights_only=True, map_location=torch.device('cpu')))

# Replace the final linear layer with a new one (1000 neurons for feature extraction)
model.fc = torch.nn.Linear(num_features, 1000)

# Move the model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Model moved to {device}")
if torch.cuda.is_available():
    devNumber = torch.cuda.current_device()
    print(f"Current device number is: {devNumber}")
    devName = torch.cuda.get_device_name(devNumber)
    print(f"GPU name is: {devName}")
    
print('-' * 75)
    
# Save the final model weights
final_model_dir = 'final_model'
os.makedirs(final_model_dir, exist_ok=True)
final_model_weights_path = os.path.join(final_model_dir, 'image_model.pt')
torch.save(model.state_dict(), final_model_weights_path)
print(f"Final model weights saved to {final_model_weights_path}")

# test the feature extraction model with an example image
print("Displaying an example image and its feature vector...")
test_features, test_labels = next(iter(test_dataloader))
test_features, test_labels = test_features.to(device), test_labels.to(device)
model.eval()
with torch.no_grad():
    features = model(test_features)

img = test_features[0]
feature_vector = features[0].cpu().numpy()

# Convert the image tensor to a PIL image and display it
plt.imshow(img.permute(1, 2, 0).cpu().numpy())
plt.title(f"Feature Vector: {feature_vector[:10]}...")  # Display the first 10 elements of the feature vector
plt.show()
