# Import the necessary libraries
from PIL import Image
import os
import sys


def clean_image_data():
    print('\n\n############## Cleaning image data ##############\n') 
    source_folder = 'source_data/images'
    target_folder = 'cleaned_data/cleaned_images'
    final_size = 512
    
    no_of_images = len(os.listdir(source_folder))
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    print(f'----> Target folder created successfully: {target_folder}\n')
    
    current_image = 0
    for image in os.listdir(source_folder):
        img_path = os.path.join(source_folder, image)
        img = Image.open(img_path)
        
        # Convert image to 'RGB' mode if it is in 'P' mode
        if img.mode in ('P', 'RGBA'):
            img = img.convert('RGB')
            
        img = img.resize((final_size, final_size))
        
        # Ensure the target file has .jpg extension
        target_image_path = os.path.join(target_folder, os.path.splitext(image)[0] + '.jpg')
        img.save(target_image_path, 'JPEG')
        
        current_image += 1
        progress(current_image, no_of_images)
    print(f'----> Images resized and saved successfully in the target folder: {target_folder}\n')
    
    print('----> Image data cleaning completed successfully\n')
    return

def progress(count, total):
        '''
        Displays a progress bar with % completed and the number 
        of processed items and total number of items to process
        
        Parameters:
        -----------
        count: number
            Number of current item to process
        total: number
            Number of total items to process
        '''
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '#' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write(f'[{bar}] {percents}%  [{count} / {total}]\r')
        sys.stdout.flush()
        
        
        
clean_image_data()  # Call the function to clean the image data

    




