import sys
import argparse
from tqdm import tqdm
from pathlib import Path

from database import insert_data
from kf_extractor import load_model, encode_image
from schemas import FeatureData, MetadataFeature

def parse_args():
    parser = argparse.ArgumentParser(description="Index image features into Milvus")
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing images to index')
    parser.add_argument('--model-name', type=str, default='ViT-H-14', help='CLIP model name')
    parser.add_argument('--pretrained', type=str, default='laion2b_s32b_b79k', help='Pretrained weights')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    return parser.parse_args()


def index_images(data_dir: str, model_name: str, pretrained: str, device: str):
    model, preprocess, _ = load_model(model_name, pretrained)
    image_paths = list(Path(data_dir).rglob('*.*'))

    data_to_insert = []

    for img_path in tqdm(image_paths, total=len(image_paths), desc="Indexing images"):
        image_features = encode_image(model, preprocess, img_path, device).cpu().numpy().flatten().tolist()
        metadata = MetadataFeature(
            class_name=img_path.parent.name,
            file_path=str(img_path)
        )
        vector_data = FeatureData(
            id=Path(img_path).stem,
            vector=image_features,
            metadata=metadata
        )
        data_to_insert.append(vector_data)

    insert_data(data_to_insert)

    print(f"Indexed {len(data_to_insert)} images from {data_dir} into Milvus.")    

if __name__ == "__main__":
    args = parse_args()
    index_images(args.data_dir, args.model_name, args.pretrained, args.device)
