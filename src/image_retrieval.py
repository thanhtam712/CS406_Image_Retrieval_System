import argparse
from pathlib import Path

from kf_extractor import load_model, encode_image, img_img_probs
from database import insert_data, delete_data, search_vector

def parse_args():
    parser = argparse.ArgumentParser(description="Image Retrieval with CLIP and Milvus")
    parser.add_argument('--image', type=str, required=True, help='Path to the image')
    parser.add_argument('--model_name', type=str, default='ViT-H-14', help='CLIP model name')
    parser.add_argument('--pretrained', type=str, default='laion2b_s32b_b79k', help='Pretrained weights')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    return parser.parse_args()

def search_image(image_path: str, model_name: str, pretrained: str, device: str, top_k: int =10):
    model, preprocess, _ = load_model(model_name, pretrained)
    image_features = encode_image(model, preprocess, image_path, device).cpu().numpy().flatten().tolist()

    results = search_vector(image_features, top_k=top_k)

    return results

if __name__ == "__main__":
    args = parse_args()
    results = search_image(args.image, args.model_name, args.pretrained, args.device)

    print(f"Top {len(results[0])} similar images for {args.image}:")
    for result in results[0]:
        print(f"Score: {result.distance}, Metadata: {result.entity.get('metadata')}")
