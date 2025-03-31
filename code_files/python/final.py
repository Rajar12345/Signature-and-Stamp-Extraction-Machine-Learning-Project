import cv2
import numpy as np
import os
import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from ultralytics import YOLO


model = YOLO('C:/Users/dasar/Desktop/HackaThon/models/best.pt')

# Perform inference on an image or folder
results = model.predict("C:/Users/dasar/Desktop/HackaThon/Test", save=True)

crop_save_dir_sign = "C:/Users/dasar/Desktop/HackaThon/python/rajDeep/cropped_signTest"
crop_save_dir_stamp = "C:/Users/dasar/Desktop/HackaThon/python/rajDeep/cropped_stampTest"
output_stamp ="C:/Users/dasar/Desktop/HackaThon/python/rajDeep/stamp_out" 
output_sign = "C:/Users/dasar/Desktop/HackaThon/python/rajDeep/sign_out" 

os.makedirs(crop_save_dir_stamp, exist_ok=True)
os.makedirs(crop_save_dir_sign, exist_ok=True)
os.makedirs(output_stamp, exist_ok=True)
os.makedirs(output_sign, exist_ok=True)

for result in results:
    image_path = result.path
    image = Image.open(image_path)

    # Iterate over each detected object
    for box in result.boxes.data:
        x1, y1, x2, y2, confidence, cls = box.tolist()

        # Get the class name of the detected object
        class_name = result.names[int(cls)]

        # Check if the detected object is a "stamp"
        if class_name == "stamp":
            # Crop the image to the bounding box
            cropped_img = image.crop((x1, y1, x2, y2))
            if cropped_img.mode in ('RGBA', 'P'):
                cropped_img = cropped_img.convert('RGB')

            # Save the cropped image with a unique name
            crop_filename = os.path.join(crop_save_dir_stamp, f"crop_{int(x1)}_{int(y1)}.jpg")
            cropped_img.save(crop_filename)

            print(f"Cropped and saved: {crop_filename}")
        if class_name == "signature":
            # Crop the image to the bounding box
            cropped_img = image.crop((x1, y1, x2, y2))
            if cropped_img.mode in ('RGBA', 'P'):
                cropped_img = cropped_img.convert('RGB')

            # Save the cropped image with a unique name
            crop_filename = os.path.join(crop_save_dir_sign, f"crop_{int(x1)}_{int(y1)}.jpg")
            cropped_img.save(crop_filename)

            print(f"Cropped and saved: {crop_filename}")


# Function to remove signature
def remove_signature(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to detect the signature (adjust threshold value as needed)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the signature
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the signature
    mask = np.zeros_like(gray)
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Inpaint the signature area using the mask
    cleaned_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return cleaned_image

# Function to reduce contrast by 50% and increase brightness by 15%
def reduce_contrast(image, contrast_factor=0.5):
    # Adjust contrast and brightness using cv2.convertScaleAbs
    # contrast_factor = 0.5 reduces contrast by 50%
    # brightness_factor = 38 increases brightness by 15% (255 * 0.15 = 38.25, rounded to 38)
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=38)
    return adjusted_image

# Main function to process images
def process_stamps(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Warning: Could not read image {filename}. Skipping...")
                continue

            # Remove signature
            cleaned_image = remove_signature(image)

            # Reduce contrast by 50% and increase brightness by 15%
            adjusted_image = reduce_contrast(cleaned_image, contrast_factor=0.5)

            # Save the processed image to the output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, adjusted_image)

            print(f"Processed and saved: {output_path}")

    print("Image processing completed for all images.")

# Input and output folders
input_folder = crop_save_dir_sign  # Replace with your input folder path
output_folder = r"C:\Users\rajde\OneDrive\Documents\h\d\16"  # Replace with your output folder path
# Process images
process_stamps(crop_save_dir_stamp, output_stamp)
