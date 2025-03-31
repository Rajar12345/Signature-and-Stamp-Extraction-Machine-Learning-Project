import cv2
import numpy as np
import os
import glob
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

def process_image(img_path, output_folder, debug_folder):
    """Processes an individual image."""
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

    except Exception as e:
        logging.error(f"Error processing {img_path}: {str(e)}")
        cv2.imwrite(os.path.join(output_folder, os.path.basename(img_path)), img)

def process_folder(input_folder, output_folder):
    """Processes all images in a folder."""
    debug_folder = os.path.join(output_folder, "debug")
    os.makedirs(output_folder, exist_ok=True)

    # Find all images in the input folder
    valid_extensions = ['.jpg', '.jpeg', '.png']  # Corrected patterns
    image_files = []
    for ext in valid_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, f"*{ext}")))

    if not image_files:
        logging.warning("No images found in input folder")
        return

    logging.info(f"Processing {len(image_files)} images...")
    for idx, img_path in enumerate(sorted(image_files), 1):
        logging.info(f"Processing image {idx}/{len(image_files)}: {os.path.basename(img_path)}")
        process_image(img_path, output_folder, debug_folder)

    logging.info("Processing complete! Check the debug folder for intermediate results")

if __name__ == "__main__":
    input_folder = r"C:\Users\dasar\Desktop\HackaThon\python\dataset\Images\batch8"
    output_folder = r"C:\Users\dasar\Desktop\HackaThon\python\dataset\Images\output"
    process_folder(input_folder, output_folder)
