import os
import cv2
import numpy as np

def remove_artifacts(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Could not read {image_path}. Skipping...")
            return None
        if img.ndim > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        thresh1 = cv2.threshold(gray, 0, 65535, cv2.THRESH_BINARY)[1]
        thresh1 = 65535 - thresh1
        count_cols = np.count_nonzero(thresh1, axis=0)
        first_x = np.where(count_cols > 0)[0][0]
        last_x = np.where(count_cols > 0)[0][-1]
        count_rows = np.count_nonzero(thresh1, axis=1)
        first_y = np.where(count_rows > 0)[0][0]
        last_y = np.where(count_rows > 0)[0][-1]
        crop = img[first_y:last_y+1, first_x:last_x+1]
        thresh2 = thresh1[first_y:last_y+1, first_x:last_x+1]
        thresh2 = 65535 - thresh2
        contours, _ = cv2.findContours(thresh2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(thresh2, dtype=np.uint8)
        cv2.drawContours(mask, [big_contour], 0, 255, -1)
        result = crop.copy()
        if result.ndim == 2 or result.shape[2] == 1:
            result[mask == 0] = 0
        else:  # Color image
            result[mask == 0] = (0, 0, 0)
        return result
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_images_recursive(source_folder, destination_folder):
    for root, dirs, files in os.walk(source_folder):
        relative_path = os.path.relpath(root, source_folder)
        current_destination = os.path.join(destination_folder, relative_path)
        if not os.path.exists(current_destination):
            os.makedirs(current_destination)

        for filename in files:
            if filename.lower().endswith('.jpg'):
                try:
                    file_path = os.path.join(root, filename)
                    result = remove_artifacts(file_path)
                    if result is not None:
                        destination_file_path = os.path.join(current_destination, filename)
                        cv2.imwrite(destination_file_path, result)
                        print(f"Processed and saved: {destination_file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}. Skipping...")

# Example usage
source_folder = './405_Extract'
destination_folder = './405_removed'

process_images_recursive(source_folder, destination_folder)
