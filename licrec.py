# Step 1: Import necessary libraries
import cv2
import pytesseract
import torch
import numpy as np
from matplotlib import pyplot as plt
import os
import requests
import subprocess
import warnings

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Step 2: Install Tesseract if not installed
if not os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
    tesseract_installer = r'C:\Users\debor\OneDrive\Desktop\Master\Semester5\Computer vision\tesseract-ocr-w64-setup-5.5.0.20241111.exe'
    if os.path.exists(tesseract_installer):
        subprocess.run([tesseract_installer, '/S'])  # Silent installation
    else:
        raise FileNotFoundError("Tesseract installer not found at the specified location.")

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Step 3: Download YOLOv5 model and set up paths
yolo_model_path = 'computer_vision/yolov5s.pt'
if not os.path.exists(yolo_model_path):
    url = 'https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt'
    response = requests.get(url)
    os.makedirs('computer_vision', exist_ok=True)
    with open(yolo_model_path, 'wb') as f:
        f.write(response.content)

script_dir = os.path.dirname(os.path.abspath(__file__))
example_images = [
    'data/plate2.jpg',
    'data/plate3.jpg',
    'data/plate4.jpg'
]

# Step 4: Load YOLO model
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path, force_reload=True)

# Step 5: Function to preprocess image and detect license plate
def detect_license_plate(image_path, model):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect license plates
    results = model(img_rgb)
    detections = results.xyxy[0]  # x1, y1, x2, y2, confidence, class

    # Debug: Print the detections
    print(f"Detections for {image_path}: {detections}")

    # Extract the region of the detected license plate and draw bounding boxes
    license_plate_img = None
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()  # Extract the values
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if int(cls) == 0:  # Assuming class 0 is 'license plate'
            license_plate_img = img_rgb[y1:y2, x1:x2]
            print(f"License plate detected at coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

    # Check if no detections were found
    if license_plate_img is None:
        print(f"No license plate detected in {image_path}")

    return img_rgb, license_plate_img

# Step 6: Function to recognize characters in the license plate
def recognize_characters(license_plate_img):
    if license_plate_img is None:
        # Return placeholders if no license plate is detected
        return "No license plate detected", None, None, None
    
    # Use pytesseract to do OCR directly on the cropped image
    custom_config = r'--oem 3 --psm 8'  # Switching to psm 8 for better single line detection
    plate_text = pytesseract.image_to_string(license_plate_img, config=custom_config)
    print(f"Recognized License Plate Text: {plate_text}")
    
    return plate_text.strip(), None, None, None

# Step 7: Iterate over example images and detect license plates
for img_path in example_images:
    # Check if the file exists
    if not os.path.exists(img_path):
        print(f"Error: File not found at {img_path}")
        continue

    # Detect license plate
    original_img, detected_plate = detect_license_plate(img_path, yolo_model)
    
    # Recognize characters
    plate_text, _, _, _ = recognize_characters(detected_plate)
    
    # Show results in matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    # Plot the original image
    axes[0].imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot the original image with bounding box
    axes[1].imshow(original_img)
    axes[1].set_title('Original Image with Bounding Box')
    axes[1].axis('off')

    # Plot the detected license plate
    if detected_plate is not None:
        axes[2].imshow(detected_plate)
        axes[2].set_title(f'Detected License Plate: {plate_text}')
    else:
        axes[2].text(0.5, 0.5, 'No license plate detected', horizontalalignment='center', verticalalignment='center', fontsize=12, color='red')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

 # Save the image with bounding box to a file
cv2.imwrite("output_image_with_bounding_box.jpg", cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
print("Image with bounding box saved as output_image_with_bounding_box.jpg")
