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

# TODO: remove the small dataset size for live version
# Use a smaller subset of the dataset for quick testing
# small_dataset_size = 100  # Adjust as necessary
# full_dataset = Subset(full_dataset, range(small_dataset_size))


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


dataloaders = {
    'train': train_dataloader,
    'val': val_dataloader
}

dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset)
}

# Load the pre-trained ResNet-50 model with the 'weights' parameter
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Get the number of input features for the final linear layer
num_features = model.fc.in_features

# Replace the final linear layer with a new one (`num_classes` is the number of categories)
num_classes = len(decoder)
model.fc = torch.nn.Linear(num_features, num_classes)

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
    
# Unfreeze the last two layers
for name, param in model.named_parameters():
    if 'layer4' in name or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
        
# Define the loss function, optimizer, and learning rate scheduler
criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Initialize TensorBoard SummaryWriter
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
model_dir = f'model_evaluation/model_{timestamp}'
os.makedirs(model_dir, exist_ok=True)
weights_dir = os.path.join(model_dir, 'weights')
os.makedirs(weights_dir, exist_ok=True)
writer = SummaryWriter(log_dir=model_dir)


# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_params_path = 'best_model_params.pt'
    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 75)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            loop_no = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                loop_no += 1
                print(f'{'-' * 25} Loop number: {loop_no} {'-' * 25}')
                print(f'Phase: {phase}, Inputs: {inputs.shape}, Labels: {labels.shape}')

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    print(f'  --> Loss: {loss.item()}')
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                print(f'  --> Running loss: {running_loss}, Running corrects: {running_corrects}')

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Log the loss and accuracy to TensorBoard
            writer.add_scalar(f'{phase} Loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase} Accuracy', epoch_acc, epoch)
            
            # Save the model weights at the end of each epoch
            epoch_weights_path = os.path.join(weights_dir, f'epoch_{epoch}_weights.pth')
            torch.save(model.state_dict(), epoch_weights_path)
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    writer.flush() # Flush the TensorBoard writer
    writer.close()  # Close the TensorBoard writer
    return model

# Train the model
model = train_model(model, criterion, optimizer, scheduler, num_epochs=25)


# Evaluate on the test dataset
model.eval()
test_running_corrects = 0

for inputs, labels in test_dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

    test_running_corrects += torch.sum(preds == labels.data)

test_acc = test_running_corrects.double() / len(test_dataset)
print(f'Test Acc: {test_acc:.4f}')

# Save test accuracy to a file
test_metrics_path = os.path.join(model_dir, 'test_metrics.txt')
with open(test_metrics_path, 'w') as f:
    f.write(f'Test Accuracy: {test_acc:.4f}\n')
    
# Display image and label for debugging
print("Displaying an example image and its predicted label...")
test_features, test_labels = next(iter(test_dataloader))
test_features, test_labels = test_features.to(device), test_labels.to(device)
outputs = model(test_features)
_, preds = torch.max(outputs, 1)

img = test_features[0]
label = test_labels[0].item()  # Convert tensor to a Python scalar
predicted_label = preds[0].item()
category_name = decoder[predicted_label]  # Use the decoder to get the category name

# Convert the image tensor to a PIL image and display it
plt.imshow(img.permute(1, 2, 0).cpu().numpy())
plt.title(f"Actual: {decoder[label]}, Predicted: {category_name}")
plt.show()