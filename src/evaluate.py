import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse
import os
import json

# Import hàm tạo model từ file classifier.py
from classifier import get_animal_classifier

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the animal classifier on Test set")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory (containing test folder)')
    parser.add_argument('--model_path', type=str, default='models/best_animal_classifier.pth', help='Path to the trained model weights')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run evaluation on')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")

    # 1. Chuẩn bị dữ liệu Test
    # LƯU Ý: Test set chỉ Resize và Crop, KHÔNG được Random Flip hay Augmentation
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dir = os.path.join(args.data_dir, 'test')
    if not os.path.exists(test_dir):
        print(f"Lỗi: Không tìm thấy thư mục test tại {test_dir}")
        return

    test_dataset = datasets.ImageFolder(test_dir, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"Evaluating on {len(test_dataset)} images of {num_classes} classes.")

    # 2. Tải mô hình
    print(f"Loading model from {args.model_path}...")
    model = get_animal_classifier(num_classes) # Tạo khung model
    
    # Nạp weights đã train vào (xử lý trường hợp load từ CPU/GPU)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval() # QUAN TRỌNG: Chuyển sang chế độ đánh giá (tắt Dropout, v.v.)

    # 3. Vòng lặp đánh giá
    running_corrects = 0
    
    # Biến để tính độ chính xác từng lớp
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))

    print("\nBắt đầu đánh giá...")
    with torch.no_grad(): # Không tính gradient để tiết kiệm bộ nhớ
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Tính tổng số đúng
            running_corrects += torch.sum(preds == labels.data)
            
            # Tính đúng cho từng lớp
            c = (preds == labels).squeeze()
            for i in range(inputs.size(0)): # Duyệt qua từng ảnh trong batch
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # 4. In kết quả chung cuộc
    total_acc = running_corrects.double() / len(test_dataset)
    print('-' * 30)
    print(f'🔥 TỔNG KẾT QUẢ TRÊN TẬP TEST:')
    print(f'👉 Overall Accuracy: {total_acc:.2%}')
    print('-' * 30)

    # 5. In kết quả chi tiết từng lớp (Optional)
    print("\nChi tiết từng lớp:")
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f' - {class_names[i]:<15s}: {acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})')
        else:
            print(f' - {class_names[i]:<15s}: N/A (no images)')

if __name__ == '__main__':
    main()