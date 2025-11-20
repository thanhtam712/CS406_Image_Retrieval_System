import os
import shutil
import random
from tqdm import tqdm

def split_dataset(source_dir, dest_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Chia bộ dataset thành 3 tập: train, val và test.
    
    Args:
        source_dir: Thư mục gốc chứa các loài động vật.
        dest_dir: Thư mục đích sẽ chứa 3 thư mục con (train, val, test).
        train_ratio: Tỉ lệ tập train (mặc định 0.8).
        val_ratio: Tỉ lệ tập val (mặc định 0.1).
        (Tập test sẽ là phần còn lại: 1.0 - train - val).
    """
    # 1. Kiểm tra tỉ lệ hợp lệ
    if train_ratio + val_ratio >= 1.0:
        print("❌ Lỗi: Tổng train_ratio và val_ratio phải nhỏ hơn 1.0 để còn chỗ cho tập test!")
        return

    test_ratio = 1.0 - (train_ratio + val_ratio)
    print(f"📊 Tỉ lệ chia: Train={train_ratio:.0%} | Val={val_ratio:.0%} | Test={test_ratio:.0%}")

    # 2. Xóa thư mục đích nếu đã tồn tại để làm mới
    if os.path.exists(dest_dir):
        print(f"🧹 Đang xóa thư mục cũ '{dest_dir}' để tạo lại...")
        shutil.rmtree(dest_dir)
    
    # 3. Tạo cấu trúc thư mục mới
    train_dir = os.path.join(dest_dir, 'train')
    val_dir = os.path.join(dest_dir, 'val')
    test_dir = os.path.join(dest_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 4. Kiểm tra thư mục nguồn
    if not os.path.exists(source_dir):
        print(f"❌ Lỗi: Không tìm thấy thư mục nguồn tại '{source_dir}'")
        return

    # Lấy danh sách các loài (các thư mục con)
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    print(f"📂 Tìm thấy {len(classes)} loài. Bắt đầu xử lý...")

    # 5. Vòng lặp xử lý từng loài
    for class_name in tqdm(classes, desc="Đang chia dữ liệu"):
        # Đường dẫn tới thư mục loài gốc
        src_class_path = os.path.join(source_dir, class_name)
        
        # Tạo thư mục loài tương ứng trong train/val/test
        dst_train_class = os.path.join(train_dir, class_name)
        dst_val_class = os.path.join(val_dir, class_name)
        dst_test_class = os.path.join(test_dir, class_name)
        
        os.makedirs(dst_train_class, exist_ok=True)
        os.makedirs(dst_val_class, exist_ok=True)
        os.makedirs(dst_test_class, exist_ok=True)

        # Lấy tất cả ảnh và xáo trộn
        images = [f for f in os.listdir(src_class_path) if os.path.isfile(os.path.join(src_class_path, f))]
        random.shuffle(images)

        # Tính toán số lượng ảnh cho mỗi tập
        count = len(images)
        train_count = int(count * train_ratio)
        val_count = int(count * val_ratio)
        # Test lấy phần còn lại để đảm bảo không sót ảnh nào do làm tròn
        
        # Chia danh sách ảnh
        train_imgs = images[:train_count]
        val_imgs = images[train_count : train_count + val_count]
        test_imgs = images[train_count + val_count :]

        # Hàm copy file cho gọn
        def copy_files(file_list, dst_folder):
            for img in file_list:
                shutil.copy(os.path.join(src_class_path, img), 
                            os.path.join(dst_folder, img))

        # Thực hiện copy
        copy_files(train_imgs, dst_train_class)
        copy_files(val_imgs, dst_val_class)
        copy_files(test_imgs, dst_test_class)

    print(f"\n✅ Xong! Dữ liệu đã được lưu tại: {dest_dir}")
    print(f"   - Train: {train_dir}")
    print(f"   - Val:   {val_dir}")
    print(f"   - Test:  {test_dir}")

if __name__ == '__main__':
    # ================= CẤU HÌNH =================
    # Đường dẫn thư mục gốc chứa 90 loài (CHỈNH LẠI NẾU CẦN)
    SOURCE_PATH = "src/animals/animals" 
    
    # Đường dẫn thư mục đầu ra
    DEST_PATH = "src/animal_dataset_split"
    
    # Tỉ lệ chia (Train - Val - Test)
    # Mặc định: 0.8 - 0.1 - 0.1
    # ============================================
    
    split_dataset(SOURCE_PATH, DEST_PATH, train_ratio=0.8, val_ratio=0.1)