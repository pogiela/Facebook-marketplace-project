import torch
from torchvision import transforms
from PIL import Image
import sys

# Define image transformations (same as those used during training)
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),  # Optional data augmentation
    transforms.RandomRotation(10),      # Optional data augmentation
    transforms.ToTensor(),              # Convert image to tensor
])

def process_image(image_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB format

    # Apply the transformations
    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)  # Add a batch dimension

    return image

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python image_processor.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    processed_image = process_image(image_path)
    
    print(f"Processed image shape: {processed_image.shape}")

    # Optionally, save the processed image tensor for verification
    torch.save(processed_image, 'processed_image.pt')
    print("Processed image tensor saved as 'processed_image.pt'")
