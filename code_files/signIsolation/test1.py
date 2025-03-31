import cv2
import numpy as np
import os
import glob
import logging
from skimage import measure, morphology
from skimage.measure import regionprops
import matplotlib.pyplot as plt

# Set up logging
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
    valid_extensions = ['.jpg', '.jpeg', '*.png']
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
    input_folder = r"C:\Users\dasar\Desktop\HackaThon\python\datasetgt\Images\batch8"    
    output_folder = r"C:\Users\dasar\Desktop\HackaThon\python\dataset\Images\output"
    
    process_folder_combined(input_folder, output_folder)
