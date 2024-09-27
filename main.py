import cv2
import torch
from torch import nn
import numpy as np
import os
from torchvision import transforms
from PIL import Image

class MNISTModel(nn.Module):
    """
    This model uses the TinyVGG architecture
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, 
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

def load_model(model_path, device):
    """
    Load the trained MNIST model.
    """
    class_names = [str(i) for i in range(10)]
    model = MNISTModel(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval() 
    return model, class_names

def main(image_path, model_path, output_folder, min_contour_area, debug=False):
    """
    Processes an input image to detect and classify digits using a pre-trained model.

    Args:
        image_path (str): Path to the input image file.
        model_path (str): Path to the pre-trained model file.
        output_folder (str): Directory where the processed images and results will be saved.
        min_contour_area (int): Minimum area threshold for contours to be considered as potential digits.
        debug (bool, optional): If True, intermediate images of detected and resized digits will be saved in the output folder.

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(model_path, device)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found at {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    min_component_area = 50  

    cleaned_thresh = np.zeros_like(thresh)

    for i in range(1, num_labels): 
        if stats[i, cv2.CC_STAT_AREA] >= min_component_area:
            cleaned_thresh[labels == i] = 255

    thresh = cleaned_thresh
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h
        
        if area > min_contour_area and 0.2 < aspect_ratio < 1.2:
            bounding_boxes.append((x, y, w, h))
    
    bounding_boxes = sorted(bounding_boxes, key=lambda box: box[0])
    digit_predictions = []
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    for idx, (x, y, w, h) in enumerate(bounding_boxes):
        digit = thresh[y:y+h, x:x+w]
        
        if debug:
            cv2.imwrite(os.path.join(output_folder, f"digit_{idx}.png"), digit)
        
        resized_digit = np.zeros((28, 28), dtype=np.uint8)
        
        aspect_ratio = w / h
        if aspect_ratio > 1:
            new_w = 20
            new_h = int(20 / aspect_ratio)
        else:
            new_h = 20
            new_w = int(20 * aspect_ratio)
        
        digit_resized = cv2.resize(digit, (new_w, new_h))
        x_pad = (28 - new_w) // 2
        y_pad = (28 - new_h) // 2
        
        resized_digit[y_pad:y_pad+new_h, x_pad:x_pad+new_w] = digit_resized
        
        if debug:
            cv2.imwrite(os.path.join(output_folder, f"digit_{idx}_.png"), resized_digit)
        
        pil_digit = Image.fromarray(resized_digit)
        digit_tensor = transform(pil_digit).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(digit_tensor)
            _, pred = torch.max(output, 1)
            predicted_digit = class_names[pred.item()]
            digit_predictions.append({
                'bounding_box': (x, y, w, h),
                'prediction': predicted_digit
            })
    
    for digit_info in digit_predictions:
        x, y, w, h = digit_info['bounding_box']
        pred = digit_info['prediction']
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, pred, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 2)
    
    result_image_path = os.path.join(output_folder, "result.png")
    cv2.imwrite(result_image_path, image)
    
    predictions = [prediction['prediction'] for prediction in digit_predictions]

    print(f"Predictions: {predictions}")
    print(f"Result image saved at {result_image_path}")

if __name__ == "__main__":
    image_path = 'sample_image.jpg'       
    model_path = './models/MNIST_model.pth'     
    output_folder = 'extracted_digits'
    min_contour_area = 500              
    debug = False                     

    main(image_path, model_path, output_folder, min_contour_area, debug)
