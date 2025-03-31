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

# Function to display masks on the image
def display_predictions(image_path, boxes, masks, labels, save_path=None):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for i, (box, mask) in enumerate(zip(boxes, masks)):
        color = np.random.randint(0, 255, (3,), dtype=int).tolist()
        
        # Draw bounding box
        cv2.rectangle(image, 
                      (int(box[0]), int(box[1])), 
                      (int(box[2]), int(box[3])), 
                      color, 2)
        
        # Apply mask with transparency
        mask = mask[0] > 0.5
        image[mask] = ((0.5 * np.array(color)) + (0.5 * image[mask])).astype(np.uint8)
    
    #plt.figure(figsize=(10, 8))
    #plt.imshow(image)
    #plt.axis('off')
    #plt.show()

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"Saved image with predictions: {save_path}")

# Iterate through images in the directory and predict
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    boxes, masks, labels = get_predictions(image_path, model, threshold=0.8)
    
    # Save and display the image with predictions
    save_path = os.path.join(output_dir, f"pred_{image_name}")
    display_predictions(image_path, boxes, masks, labels, save_path)
