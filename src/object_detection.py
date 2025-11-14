import cv2
import os
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Detect and crop animals from an image")
    parser.add_argument('--input_image', type=str, default="synthetic_images/synthetic_coyote_flamingo.jpg", help='Path to the input image')
    parser.add_argument('--output_dir', type=str, default="synthetic_images/cropped_animals", help='Directory to save cropped images')
    return parser.parse_args()

def detect_and_crop_animals(input_image, output_dir, conf_threshold=0.6):
    """
    Detect animals in an image and crop them to separate files.
    
    Args:
        input_image: Path to input image
        output_dir: Directory to save cropped images
        conf_threshold: Minimum confidence threshold (default: 0.5)
    """
    model = YOLO("models/yolo11s.pt")
    
    results = model(input_image)
    result = results[0]
    
    img = cv2.imread(input_image)
    
    os.makedirs(output_dir, exist_ok=True)
    
    cropped_paths = []
    
    num_objects = len(result.boxes)
    
    # If no objects detected, return empty list
    if num_objects == 0:
        print("No objects detected in the image")
        return cropped_paths
    
    should_crop = num_objects > 1
    
    for idx, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = result.names[cls]
        
        # Skip detections below confidence threshold
        if conf < conf_threshold:
            print(f"Skipping {class_name} with low confidence: {conf:.2f}")
            continue
        
        if should_crop:
            cropped_img = img[y1:y2, x1:x2]
            crop_filename = f"{output_dir}/crop_{idx}.jpg"
            cv2.imwrite(crop_filename, cropped_img)
            cropped_paths.append(crop_filename)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"{class_name}: {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    if not should_crop:
        box = result.boxes[0]
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = result.names[cls]
        output_image_path = f"{output_dir}/crop_0.jpg"
        cv2.imwrite(output_image_path, img)
        cropped_paths.append(output_image_path)
        print(f"Detected 1 object - saved original image to: {output_image_path}")
    
    return cropped_paths


if __name__ == "__main__":
    args = parse_args()
    input_image = args.input_image
    output_dir = args.output_dir
    
    cropped_paths = detect_and_crop_animals(input_image, output_dir)
    
    print(f"\nCropped {len(cropped_paths)} images:")
    for path in cropped_paths:
        print(f"  - {path}")
