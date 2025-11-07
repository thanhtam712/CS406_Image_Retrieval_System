import torch
import json
import argparse
from PIL import Image
from torchvision import transforms

from classifier import load_trained_classifier

def parse_args():
    parser = argparse.ArgumentParser(description="Classify an animal image")
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model_path', type=str, default='models/best_animal_classifier.pth', help='Path to the trained model weights')
    parser.add_argument('--class_names_path', type=str, default='models/class_names.json', help='Path to the class names JSON file')
    parser.add_argument('--summaries_path', type=str, default='src/animal_summaries.json', help='Path to the animal summaries JSON file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on (cuda or cpu)')
    return parser.parse_args()

def predict_image(image_path: str, model, class_names: list, device: str):
    """
    Dự đoán lớp của một ảnh đầu vào.
    """
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_catid = torch.max(probabilities, 0)
    
    predicted_class = class_names[top_catid.item()]
    confidence = top_prob.item()
    
    return predicted_class, confidence

def main():
    args = parse_args()
    
    # Tải thông tin cần thiết
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    with open(args.class_names_path, 'r') as f:
        class_names = json.load(f)
        
    with open(args.summaries_path, 'r', encoding='utf-8') as f:
        summaries = json.load(f)
        
    num_classes = len(class_names)
    
    # Tải mô hình đã huấn luyện
    model = load_trained_classifier(args.model_path, num_classes, device)
    
    # Dự đoán
    predicted_class, confidence = predict_image(args.image, model, class_names, device)
    
    # Lấy tóm tắt và in kết quả
    summary = summaries.get(predicted_class, "Không có thông tin tóm tắt cho loài này.")
    
    print("--- NHÁNH 2: PHÂN LOẠI & TÓM TẮT (CNN Tự train) ---")
    print(f"Output Class:    {predicted_class}")
    print(f"Confidence:      {confidence:.2%}")
    print(f"Output Summarize: {summary}")

if __name__ == '__main__':
    main()