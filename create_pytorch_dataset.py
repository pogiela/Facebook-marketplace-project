import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
import matplotlib.pyplot as plt
import pickle


class CustomImageDataset(Dataset):    
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1] + '.jpg')
        image = read_image(img_path).float() / 255.0  # Convert images to floating-point tensors and normalize to [0, 1]
        
        # Check the number of channels and convert grayscale to RGB
        if image.shape[0] == 1:  # Grayscale image
            image = image.repeat(3, 1, 1)  # Repeat the single channel 3 times

        label = self.img_labels.iloc[idx, 3]
        
        if self.transform:
            image = self.transform(image)
            
        return (image, torch.tensor(label))
    

if __name__ == "__main__":
    # Variables
    file_path = 'cleaned_data/training_data.csv'
    img_dir = "cleaned_data/images/"

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),  # Optional data augmentation
        transforms.RandomRotation(10),      # Optional data augmentation
    ])

    # Create a custom dataset
    training_data = CustomImageDataset(annotations_file=file_path, img_dir=img_dir, transform=transform) 
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

    # Load decoder dictionary from file
    with open("image_decoder.pkl", "rb") as f:
        decoder_dict = pickle.load(f)

    # Extract the encoder and decoder from the loaded dictionary
    encoder = decoder_dict['encoder']
    decoder = decoder_dict['decoder']


    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0]
    label = train_labels[0]
    category_name = decoder[label.item()]  # Convert tensor to a Python scalar and use the decoder to get the category name
    print(f"Label: {label} - {category_name}")

    # Convert the image tensor to a PIL image and display it
    plt.imshow(img.permute(1, 2, 0))
    plt.title(f"Label: {label} - {category_name}")
    plt.show()

    # for image in train_features:
    #     print(image.shape)