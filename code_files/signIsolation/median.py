import os
import cv2

# Define the input and output directories
input_dir = r"C:\Users\dasar\Desktop\HackaThon\python\dataset\Images\batch7"  # Replace with your input folder path
output_dir = 'C:/path/to/output_folder'  # Replace with your output folder path
os.makedirs(output_dir, exist_ok=True)

# Define the kernel size for the median filter
kernel_size = 5  # Must be an odd integer

# Iterate through all image files in the input directory
for image_name in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_name)
    
    # Check if the file is an image
    if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        # Read the image
        image = cv2.imread(image_path)
        
        # Apply median filter
        filtered_image = cv2.medianBlur(image, kernel_size)
        
        # Save the filtered image to the output directory
        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, filtered_image)
        
        print(f'Processed and saved: {output_path}')
    else:
        print(f'Skipped non-image file: {image_name}')

print('All images processed successfully!')

