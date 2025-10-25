import torch
import open_clip

from PIL import Image
from pathlib import Path

def load_model(model_name: str ='ViT-H-14', pretrained: str ='laion2b_s32b_b79k', device: str = 'cuda'):
    """Load a pre-trained CLIP model and its preprocessing transforms."""
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, cache_dir="./cache_dir")
    model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    return model, preprocess, tokenizer

def encode_image(model: torch.nn.Module, preprocess, image_path: Path | str, device: str ='cuda'):
    """Encode an image to its CLIP feature representation."""
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad(), torch.autocast(device):
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features

def img_img_probs(image1_features: torch.Tensor, image2_features: torch.Tensor):
    """Compute similarity probabilities between two sets of image features."""
    with torch.no_grad():
        probs = (100.0 * image1_features @ image2_features.T).softmax(dim=-1)
    return probs
