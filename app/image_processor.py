'''
image_processor.py

This module contains the function to process an image for inference in a machine learning model.

The following steps are performed:
1. Load the image from the specified path.
2. Apply a series of transformations to prepare the image for the model:
   - Convert the image to a tensor.
   - Resize the image.
   - Center crop the image to the final size.
   - Normalize the image using mean and standard deviation.
3. Add a batch dimension to the processed image tensor, as models typically expect batched input.

Usage:
    To process an image, run the script with the image path as an argument:
    $ python image_processor.py <image_path>
'''
import torch
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
    """
    Processes an image for model inference.

    Args:
        image_path (str): Path to the image file to be processed.

    Returns:
        torch.Tensor: Processed image tensor with shape [1, 3, 224, 224].
    """
    # Load the image
    image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB format

    # Apply the transformations
    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)  # Add a batch dimension

    return image

if __name__ == "__main__":
    # Ensure the correct usage of the script
    if len(sys.argv) != 2:
        print("Usage: python image_processor.py <image_path>")
        sys.exit(1)

    # Get the image path from command-line arguments
    image_path = sys.argv[1]
    
    # Process the image
    processed_image = process_image(image_path)
    
    # Display the shape of the processed image tensor
    print(f"Processed image shape: {processed_image.shape}")

    # Optionally, save the processed image tensor for verification
    torch.save(processed_image, 'processed_image.pt')
    print("Processed image tensor saved as 'processed_image.pt'")