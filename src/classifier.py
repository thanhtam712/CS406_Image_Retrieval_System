import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name, num_classes, device, pretrained=True):
    """
    Hàm factory chọn model.
    Hỗ trợ: 'resnet50', 'mobilenet', 'vgg16'

    Args:
        model_name (str): Tên model ('resnet50', 'mobilenet', 'vgg16')
        num_classes (int): Số lượng lớp (90 loài)
        device (torch.device): CPU hoặc CUDA
        pretrained (bool): Có dùng weights đã học trên ImageNet không
    """
    model_name = model_name.lower()

    if model_name == 'resnet50':
        # 1. ResNet50
        print(f"Initializing ResNet50...")
        # Tải kiến trúc ResNet50
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)

        # Đóng băng (Freeze) các lớp feature extraction
        if pretrained:
            for param in model.parameters():
                param.requires_grad = False

        # Thay thế lớp cuối (fc)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'mobilenet':
        # 2. MobileNetV2
        print(f"Initializing MobileNetV2...")
        # Tải kiến trúc MobileNetV2
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v2(weights=weights)

        # Đóng băng
        if pretrained:
            for param in model.parameters():
                param.requires_grad = False

        # MobileNet lớp cuối nằm ở: classifier[1]
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

        # Gán lớp này vào biến 'fc' để file train.py không bị lỗi
        # vì file train đang gọi optimizer(model.fc.parameters()...)
        model.fc = model.classifier[1]

    elif model_name == 'vgg16':
        # 3. VGG16
        print(f"Initializing VGG16...")
        # Tải kiến trúc VGG16
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg16(weights=weights)

        # Đóng băng
        if pretrained:
            for param in model.parameters():
                param.requires_grad = False

        # VGG16 lớp cuối nằm ở: classifier[6]
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

        # Hack tên 'fc' cho tương thích file train
        model.fc = model.classifier[6]
    
    else:
        raise ValueError(f"Model {model_name} chưa được hỗ trợ! Hãy chọn: resnet50, mobilenet, vgg16")
    return model.to(device)

def load_trained_classifier(model_path: str, model_name: str, num_classes: int, device: str):
    """
    Tải mô hình đã được huấn luyện từ một file .pth.
    """
    # 1. Gọi hàm get_model
    # pretrained=False vì ta sắp load weights của riêng mình vào rồi
    model = get_model(model_name, num_classes, device, pretrained=False)

    # 2. Load weights
    # map_location giúp tránh lỗi khi train bằng GPU nhưng load bằng CPU (hoặc ngược lại)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 3. Chuyển sang device
    model.to(device)
    
    # 4. Chế độ đánh giá
    model.eval() 

    return model