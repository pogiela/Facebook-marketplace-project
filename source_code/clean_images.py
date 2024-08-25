'''
This is a module containing functions to load and clean image data

Functions:
    clean_image_data: Function to clean the image data
'''
from functions import progress
from PIL import Image
import os
import sys

########### VARIABLES ###########
final_size = 512

########### FUNCTIONS ###########
def resize_image(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.Resampling.LANCZOS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im


def clean_image_data(source_folder, target_folder):
    '''
    Function to clean the image data
    '''
    print('\n\n############## Cleaning image data ##############\n') 
    no_of_images = len(os.listdir(source_folder))
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    print(f'----> Target folder created successfully: {target_folder}\n')
    
    current_image = 0
    for image in os.listdir(source_folder):
        img_path = os.path.join(source_folder, image)
        img = Image.open(img_path)
        
        # Convert image to 'RGB' mode if it is in 'P' mode
        # if img.mode in ('P', 'RGBA'):
        #     img = img.convert('RGB')
        
        new_im = resize_image(final_size, img)
        
        # Ensure the target file has .jpg extension
        target_image_path = os.path.join(target_folder, os.path.splitext(image)[0] + '.jpg')
        new_im.save(target_image_path, 'JPEG')
        
        current_image += 1
        progress(current_image, no_of_images)
        
    print(f'\n----> Images resized and saved successfully in the target folder: {target_folder}\n')
    
    print('----> Image data cleaning completed successfully\n')
    return
        
if __name__ == "__main__":
    print('This is a module containing functions to clean images')
    print('Please run the main programme to execute the data processing')
    sys.exit()
