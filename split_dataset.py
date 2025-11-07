# file: split_dataset.py

import os
import shutil
import random
from tqdm import tqdm

def split_dataset(source_dir, dest_dir, split_ratio=0.8):
    """
    Chia bộ dataset thành các tập train và val.

    Args:
        source_dir (str): Đường dẫn đến thư mục gốc chứa 90 thư mục loài.
        dest_dir (str): Đường dẫn đến thư mục mới để chứa 'train' và 'val'.
        split_ratio (float): Tỉ lệ dữ liệu cho tập train (ví dụ: 0.8 là 80%).
    """
    # Xóa thư mục đích nếu đã tồn tại để làm lại từ đầu
    if os.path.exists(dest_dir):
        print(f"Thư mục đích '{dest_dir}' đã tồn tại. Đang xóa để tạo lại...")
        shutil.rmtree(dest_dir)
    
    # Tạo các thư mục train và val
    train_path = os.path.join(dest_dir, 'train')
    val_path = os.path.join(dest_dir, 'val')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    
    # Lấy danh sách các thư mục lớp (tên loài động vật)
    class_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    print(f"Bắt đầu chia {len(class_dirs)} lớp từ '{source_dir}' vào '{dest_dir}'...")
    
    # Lặp qua từng thư mục lớp
    for class_name in tqdm(class_dirs, desc="Processing classes"):
        class_source_path = os.path.join(source_dir, class_name)
        
        # Tạo thư mục con tương ứng trong train và val
        os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_path, class_name), exist_ok=True)
        
        # Lấy danh sách tất cả các ảnh
        images = [f for f in os.listdir(class_source_path) if os.path.isfile(os.path.join(class_source_path, f))]
        random.shuffle(images) # Xáo trộn ngẫu nhiên
        
        # Tính toán điểm chia
        split_point = int(len(images) * split_ratio)
        
        # Chia danh sách ảnh
        train_images = images[:split_point]
        val_images = images[split_point:]
        
        # Sao chép file vào thư mục train
        for image_name in train_images:
            shutil.copy(
                os.path.join(class_source_path, image_name),
                os.path.join(train_path, class_name, image_name)
            )
            
        # Sao chép file vào thư mục val
        for image_name in val_images:
            shutil.copy(
                os.path.join(class_source_path, image_name),
                os.path.join(val_path, class_name, image_name)
            )
            
    print("Hoàn tất việc chia dataset!")


if __name__ == '__main__':
    # ================== BẠN CHỈ CẦN CHỈNH 2 DÒNG DƯỚI ĐÂY ==================
    
    # 1. Đặt đường dẫn đến thư mục chứa 90 thư mục động vật của bạn
    SOURCE_DIRECTORY = "src/animals/animals" # Ví dụ: tên thư mục gốc là "animals"
    
    # 2. Đặt tên cho thư mục mới sẽ chứa kết quả sau khi chia
    DESTINATION_DIRECTORY = "src/animal_dataset" # Thư mục này sẽ được tự động tạo
    
    # ======================================================================
    
    split_dataset(SOURCE_DIRECTORY, DESTINATION_DIRECTORY, split_ratio=0.8) # 80% train, 20% val