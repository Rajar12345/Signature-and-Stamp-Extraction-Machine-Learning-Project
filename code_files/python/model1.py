import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from ultralytics import YOLO
# Load the model
save_path = "C:/Users/dasar/Desktop/HackaThon/python/predict"
os.makedirs(save_path, exist_ok=True)

model = YOLO('C:/Users/dasar/Desktop/HackaThon/models/best.pt')

# Perform inference on an image or folder
results = model.predict("C:/Users/dasar/Desktop/HackaThon/VIT_Dataset", save=True)

crop_save_dir = "C:/Users/dasar/Desktop/HackaThon/python/cropped_signatures"
os.makedirs(crop_save_dir, exist_ok=True)

os.makedirs(crop_save_dir, exist_ok=True)
for result in results:
    image_path = result.path
    image = Image.open(image_path)

    # Iterate over each detected object
    for box in result.boxes.data:
        x1, y1, x2, y2, confidence, cls = box.tolist()

        # Get the class name of the detected object
        class_name = result.names[int(cls)]

        # Check if the detected object is a "stamp"
        if class_name == "signature":
            # Crop the image to the bounding box
            cropped_img = image.crop((x1, y1, x2, y2))
            if cropped_img.mode in ('RGBA', 'P'):
                cropped_img = cropped_img.convert('RGB')

            # Save the cropped image with a unique name
            crop_filename = os.path.join(crop_save_dir, f"crop_{int(x1)}_{int(y1)}.jpg")
            cropped_img.save(crop_filename)

            print(f"Cropped and saved: {crop_filename}")
