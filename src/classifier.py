import torch
import torch.nn as nn
from torchvision import models

def get_animal_classifier(num_classes: int, pretrained: bool = True):
    # Tải kiến trúc ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def load_trained_classifier(model_path: str, num_classes: int, device: str):
    """
    Tải mô hình đã được huấn luyện từ một file .pth.
    """
    model = get_animal_classifier(num_classes, pretrained=False) 
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() 
    return model