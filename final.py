import cv2
import numpy as np
import os
import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from ultralytics import YOLO


import glob
import logging
from skimage import measure, morphology
from skimage.measure import regionprops
import matplotlib.pyplot as plt


model = YOLO('./best.pt')

# Perform inference on an image or folder
results = model.predict("C:/Users/dasar/Desktop/HackaThon/VIT_Dataset", save=True)

crop_save_dir_sign = "./cropped_signTest"
crop_save_dir_stamp = "./cropped_stamps"
output_stamp ="./stamp_out" 
output_sign = "./sign_out" 

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
# Process images
process_stamps(crop_save_dir_stamp, output_stamp)
#################################################################################################################################################################


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants for the second script
constant_parameter_1 = 30
constant_parameter_2 = 80
constant_parameter_3 = 30
constant_parameter_4 = 8

def debug_save(image, path, name):
    """Helper function to save intermediate images for debugging."""
    cv2.imwrite(os.path.join(path, f"debug_{name}.jpg"), image)

def remove_stamps(img, debug_path):
    """Removes blue stamps using HSV filtering and inpainting."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the blue color range for detection
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    debug_save(blue_mask, debug_path, "1_blue_mask")

    # Refine the mask to avoid affecting the signature
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    debug_save(blue_mask, debug_path, "2_refined_blue_mask")

    # Apply inpainting to remove stamps
    inpainted = cv2.inpaint(img, blue_mask, 3, cv2.INPAINT_TELEA)
    debug_save(inpainted, debug_path, "3_stamp_removed")

    return inpainted

def extract_signature(img, debug_path):
    """Extracts black signatures while filtering out other elements."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    debug_save(gray, debug_path, "4_gray_image")

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    debug_save(enhanced, debug_path, "5_clahe_enhanced")

    # Adaptive thresholding to extract black regions (signatures)
    block_size = max(15, (min(gray.shape) // 8) | 1)  # Ensure block size is odd
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        7
    )
    debug_save(binary, debug_path, "6_adaptive_thresh")

    # Morphological operations to refine the mask
    kernel = np.ones((3,3), np.uint8)
    refined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)
    debug_save(refined, debug_path, "7_morph_processed")

    return refined

def adjust_output(img):
    """Reduces brightness, increases contrast, and decreases exposure."""
    brightness_factor = 0.5  # Reduce brightness by 50%
    contrast_factor = 1.5    # Increase contrast by 50%
    exposure_factor = 0.7    # Decrease exposure

    img = img.astype(np.float32)
    img = img * contrast_factor  # Increase contrast
    img = img * exposure_factor  # Decrease exposure
    img = img - (255 * (1 - brightness_factor))  # Reduce brightness
    img = np.clip(img, 0, 255)  # Ensure valid range

    return img.astype(np.uint8)

def preprocess_image(img):
    """Preprocesses the image for the second script."""
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Handle completely black or white images
    if np.all(img == img[0, 0]):
        raise ValueError("Image is completely blank or single-color")
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

def safe_threshold(img):
    """Applies thresholding safely."""
    try:
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    except Exception as e:
        print(f"Thresholding error: {e}")
        img = np.zeros_like(img)
    return img

def process_image_combined(img_path, output_folder, debug_folder):
    """Processes an individual image with combined logic."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Could not read image")

        os.makedirs(debug_folder, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        img_debug_path = os.path.join(debug_folder, base_name)
        os.makedirs(img_debug_path, exist_ok=True)

        # Remove blue stamps
        cleaned = remove_stamps(img, img_debug_path)

        # Extract black signature
        signature_mask = extract_signature(cleaned, img_debug_path)
        debug_save(signature_mask, img_debug_path, "8_signature_mask")

        # Use the mask to extract the signature
        signature = cv2.bitwise_and(cleaned, cleaned, mask=signature_mask)

        # Create a white background
        white_bg = np.full_like(cleaned, 255)

        # Merge signature onto white background
        final = np.where(signature_mask[..., None] == 255, signature, white_bg)

        # Apply brightness, contrast & exposure adjustments only to output
        final_adjusted = adjust_output(final)
        debug_save(final_adjusted, img_debug_path, "9_final_adjusted_output")

        # Save the final output
        output_path = os.path.join(output_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, final_adjusted)
        logging.info(f"Processed: {output_path}")

        # Now apply the second script's logic
        img = preprocess_image(final_adjusted)
        img = safe_threshold(img)

        blobs = img > img.mean()
        blobs_labels = measure.label(blobs, background=1)

        total_area, counter, biggest_component = 0, 0, 0
        for region in regionprops(blobs_labels):
            if region.area > 10:
                total_area += region.area
                counter += 1
                if region.area > biggest_component:
                    biggest_component = region.area
        
        if counter == 0:
            raise ValueError(f"No valid regions found in {img_path}")
        
        average = total_area / counter
        a4_small_size_outliar_constant = ((average / constant_parameter_1) * constant_parameter_2) + constant_parameter_3
        a4_big_size_outliar_constant = a4_small_size_outliar_constant * constant_parameter_4

        pre_version = morphology.remove_small_objects(blobs_labels, a4_small_size_outliar_constant)
        component_sizes = np.bincount(pre_version.ravel())
        
        # Handle edge case: if no components are large enough
        if len(component_sizes) == 0 or component_sizes.max() == 0:
            raise ValueError("No significant components found after filtering")

        too_small = component_sizes > a4_big_size_outliar_constant
        too_small_mask = too_small[pre_version]
        pre_version[too_small_mask] = 0

        plt.imsave('pre_version.png', pre_version)

        img = cv2.imread('pre_version.png', 0)
        if img is None or img.size == 0:
            raise ValueError("Error loading pre-processed image")

        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        output_path = os.path.join(output_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, img)
        logging.info(f"Saved output to {output_path}\n")

    except Exception as e:
        logging.error(f"Error processing {img_path}: {str(e)}")
        if 'img' in locals() and img is not None:
            output_path = os.path.join(output_folder, os.path.basename(img_path))
            cv2.imwrite(output_path, img)
            logging.info(f"Saved original image as fallback: {output_path}")

def process_folder_combined(input_folder, output_folder):
    """Processes all images in a folder with combined logic."""
    debug_folder = os.path.join(output_folder, "debug")
    os.makedirs(output_folder, exist_ok=True)

    # Find all images in the input folder
# Corrected pattern for all common image formats
    valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']

    image_files = []
    for ext in valid_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))

    if not image_files:
        logging.warning("No images found in input folder")
        return

    logging.info(f"Processing {len(image_files)} images...")
    for idx, img_path in enumerate(sorted(image_files), 1):
        logging.info(f"Processing image {idx}/{len(image_files)}: {os.path.basename(img_path)}")
        process_image_combined(img_path, output_folder, debug_folder)

    logging.info("Processing complete! Check the debug folder for intermediate results")

if __name__ == "__main__":
    process_folder_combined(crop_save_dir_sign, output_sign)

