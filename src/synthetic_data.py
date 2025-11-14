import os
import random
from PIL import Image
import numpy as np
from pathlib import Path


def get_animal_classes(animals_dir):
    """Get list of animal class directories."""
    return [d for d in os.listdir(animals_dir) 
            if os.path.isdir(os.path.join(animals_dir, d))]


def get_random_image_from_class(animals_dir, class_name):
    """Get a random image path from a specific animal class."""
    class_dir = os.path.join(animals_dir, class_name)
    images = [f for f in os.listdir(class_dir) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        return None
    
    return os.path.join(class_dir, random.choice(images))


def resize_to_same_size(images, target_size=(224, 224)):
    """Resize all images to the same size."""
    resized_images = []
    for img in images:
        resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
        resized_images.append(resized_img)
    return resized_images


def merge_images(image_paths, grid_size='auto'):
    """
    Merge multiple images into a single grid image.
    
    Args:
        image_paths: List of image file paths
        grid_size: 'auto' (determines based on number of images), 
                   '2x1', '1x2', '2x2', etc.
    
    Returns:
        PIL Image object with merged images
    """
    if not image_paths:
        raise ValueError("No image paths provided")
    
    # Load images
    images = [Image.open(path).convert('RGB') for path in image_paths]
    
    # Resize all images to same size
    images = resize_to_same_size(images)
    
    # Determine grid layout
    num_images = len(images)
    if grid_size == 'auto':
        if num_images == 2:
            rows, cols = 1, 2  # 1 row, 2 columns
        elif num_images == 4:
            rows, cols = 2, 2  # 2x2 grid
        else:
            # For other numbers, try to make a square-ish grid
            rows = int(np.sqrt(num_images))
            cols = int(np.ceil(num_images / rows))
    else:
        rows, cols = map(int, grid_size.split('x'))
    
    # Get dimensions
    img_width, img_height = images[0].size
    
    # Create blank canvas
    merged_width = cols * img_width
    merged_height = rows * img_height
    merged_image = Image.new('RGB', (merged_width, merged_height), (255, 255, 255))
    
    # Paste images onto canvas
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * img_width
        y = row * img_height
        merged_image.paste(img, (x, y))
    
    return merged_image


def create_synthetic_image(animals_dir, num_images=None, output_path=None, auto_name=True):
    """
    Create a synthetic image by merging random images from different animal classes.
    
    Args:
        animals_dir: Path to the animals directory
        num_images: Number of images to merge (2 or 4). If None, randomly choose.
        output_path: Where to save the merged image. If None, display only.
        auto_name: If True, automatically generate filename with animal names.
    
    Returns:
        tuple: (merged_image, selected_classes)
    """
    # Randomly choose 2 or 4 if not specified
    if num_images is None:
        num_images = random.choice([2, 4])
    
    if num_images not in [2, 4]:
        raise ValueError("num_images must be 2 or 4")
    
    # Get all animal classes
    all_classes = get_animal_classes(animals_dir)
    
    if len(all_classes) < num_images:
        raise ValueError(f"Not enough animal classes. Need {num_images}, found {len(all_classes)}")
    
    # Randomly select different classes
    selected_classes = random.sample(all_classes, num_images)
    
    # Get random images from each selected class
    image_paths = []
    for class_name in selected_classes:
        img_path = get_random_image_from_class(animals_dir, class_name)
        if img_path:
            image_paths.append(img_path)
    
    if len(image_paths) != num_images:
        raise ValueError(f"Could not find images for all selected classes")
    
    # Merge images
    merged_image = merge_images(image_paths)
    
    # Save if output path is provided
    if output_path:
        # Auto-generate filename with animal names if requested
        if auto_name:
            output_dir = os.path.dirname(output_path) or '.'
            animal_names = '_'.join(selected_classes)
            filename = f"synthetic_{animal_names}.jpg"
            output_path = os.path.join(output_dir, filename)
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        merged_image.save(output_path)
        print(f"Saved merged image to: {output_path}")
    
    return merged_image, selected_classes


def generate_multiple_synthetic_images(animals_dir, num_synthetic_images=10, 
                                       output_dir='./synthetic_images'):
    """
    Generate multiple synthetic images.
    
    Args:
        animals_dir: Path to the animals directory
        num_synthetic_images: How many synthetic images to generate
        output_dir: Directory to save the synthetic images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_synthetic_images):
        num_images = random.choice([2, 4])
        output_path = os.path.join(output_dir, f'synthetic_{i+1}.jpg')
        
        try:
            merged_image, selected_classes = create_synthetic_image(
                animals_dir, 
                num_images=num_images,
                output_path=output_path,
                auto_name=True
            )
            print(f"Created synthetic image {i+1}/{num_synthetic_images}: {', '.join(selected_classes)}")
        except Exception as e:
            print(f"Error creating synthetic image {i+1}: {e}")


if __name__ == "__main__":
    # Path to animals directory
    animals_dir = "/mlcv3/WorkingSpace/Personal/baotg/TTam/CS406/data/animals"
    
    print("\nGenerating multiple synthetic images...")
    generate_multiple_synthetic_images(
        animals_dir,
        num_synthetic_images=10,
        output_dir="synthetic_images"
    )
    
    print("\nDone!")
