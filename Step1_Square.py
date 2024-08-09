import os
from PIL import ImageOps, Image
import numpy as np

def process_and_save_image(image_path, dest_path):
    try:
        with Image.open(image_path) as img:
            img_array = np.array(img)
            
            # Determine the presence of the object in the image
            rows = np.any(img_array, axis=1)
            cols = np.any(img_array, axis=0)
            
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            cropped_img = img.crop((xmin, ymin, xmax, ymax))
            width, height = cropped_img.size
            
            # Calculate how much padding is needed to make the image square
            diff = abs(width - height)
            
            # Determine the position of the breast to decide where to add padding
            center_x = (xmax + xmin) / 2
            orig_width = img.size[0]
            
            # If breast is more to the left, add padding to the right, and vice versa
            if center_x < orig_width / 2:
                # Breast is to the left, pad on the right
                padded_img = ImageOps.expand(cropped_img, (0, 0, diff, 0), fill=0)
            else:
                # Breast is to the right, pad on the left
                padded_img = ImageOps.expand(cropped_img, (diff, 0, 0, 0), fill=0)
            
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Save the processed image
            padded_img.save(dest_path, format='TIFF')
            print(f"Processed and saved {dest_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}. Skipping...")

def process_images_recursive(src_dir, dest_dir):
    for root, dirs, files in os.walk(src_dir):
        relative_path = os.path.relpath(root, src_dir)
        current_dest_dir = os.path.join(dest_dir, relative_path)
        
        for filename in files:
            if filename.endswith(".tif"):
                src_path = os.path.join(root, filename)
                dest_path = os.path.join(current_dest_dir, filename)
                process_and_save_image(src_path, dest_path)

# Example usage
src_dir = './Data/Normal'
dest_dir = './Data_Pre/Normal'

process_images_recursive(src_dir, dest_dir)
