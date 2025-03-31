import os
import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import transforms as T
from matplotlib import pyplot as plt
from PIL import Image

# Define paths
image_dir = 'C:/Users/dasar/Desktop/HackaThon/VIT_Dataset'  # Change this to your images directory
output_dir = 'C:/Users/dasar/Desktop/HackaThon/python/predict'
os.makedirs(output_dir, exist_ok=True)

# Load the pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define image transformations
transform = T.Compose([
    T.ToTensor(),
])

# Function to get predictions from the model
def get_predictions(image_path, model, threshold=0.8):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)
    with torch.no_grad():
        predictions = model([image_tensor])
    
    # Extract relevant data from predictions
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    masks = predictions[0]['masks'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()

    # Filter out low-score results
    high_score_indices = [i for i, score in enumerate(scores) if score > threshold]
    boxes = boxes[high_score_indices]
    masks = masks[high_score_indices]
    labels = labels[high_score_indices]

    return boxes, masks, labels

# Function to crop and save images based on masks
def crop_and_save(image_path, boxes, masks, labels, save_dir):
    image = cv2.imread(image_path)
    image_name = os.path.basename(image_path).split('.')[0]
    
    for i, (box, mask) in enumerate(zip(boxes, masks)):
        # Apply mask and crop
        mask = mask[0] > 0.5
        masked_image = image * mask[:, :, np.newaxis]
        
        # Extract bounding box and crop
        x_min, y_min, x_max, y_max = map(int, box)
        cropped_image = masked_image[y_min:y_max, x_min:x_max]
        
        # Save the cropped image
        crop_filename = f"{image_name}_crop_{i}.png"
        crop_save_path = os.path.join(save_dir, crop_filename)
        cv2.imwrite(crop_save_path, cropped_image)
        print(f"Saved cropped image: {crop_save_path}")

# Iterate through images in the directory and predict
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    boxes, masks, labels = get_predictions(image_path, model, threshold=0.8)
    
    # Crop and save the detected regions
    crop_and_save(image_path, boxes, masks, labels, output_dir)

