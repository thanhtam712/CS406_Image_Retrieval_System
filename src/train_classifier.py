import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse
import os
import json

from classifier import get_animal_classifier

def parse_args():
    parser = argparse.ArgumentParser(description="Train an animal classifier")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory (containing train and val)')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save the trained model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run training on (cuda or cpu)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Chuẩn bị Data
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
    
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {', '.join(class_names)}")

    # Khởi tạo Mô hình, Loss, Optimizer
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = get_animal_classifier(num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr) 

    # Vòng lặp Training
    best_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} phase"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Lưu lại model tốt nhất
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                model_path = os.path.join(args.output_dir, 'best_animal_classifier.pth')
                torch.save(model.state_dict(), model_path)
                print(f"New best model saved to {model_path} with accuracy: {best_acc:.4f}")

    # Lưu lại tên các lớp để dùng khi predict
    class_names_path = os.path.join(args.output_dir, 'class_names.json')
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f)
    print(f"Class names saved to {class_names_path}")


if __name__ == '__main__':
    main()