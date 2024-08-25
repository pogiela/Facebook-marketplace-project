'''
This module contains the function to process an image for inference.

The following steps are performed:
1. Load the image
2. Apply the transformations
3. Add a batch dimension

'''
import torch
from torchvision import transforms
from PIL import Image
import sys
from torchvision.transforms import v2


########## VARIABLES ##########
final_size = 224    # Final size of the images

# Define image transformations (same as those used for validation during training)
transform = v2.Compose([
        v2.ToImage(), # Convert to tensor, only needed for PIL images
        v2.Resize(256),
        v2.CenterCrop(final_size),
        v2.ToDtype(torch.float32, scale=True), # this has replaced ToTensor()
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def process_image(image_path):
    '''
    Function to process an image for inference
    '''
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