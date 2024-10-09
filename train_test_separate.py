import os
import shutil
import random
from pathlib import Path

# 데이터셋 경로 설정
source_dir = r"C:\sg0117\dataset\without_augmentation"   # 원본 데이터 있는 경로
destination_dir = r"C:\sg0117\dataset\data"              # train / test 분류될 폴더 경로

# 클래스별 폴더 이름 설정
categories = {
    "healthy": "healthy",
    "diseased": "diseased",
    "background": "Background_without_leaves"
}

# 고유한 파일 이름 생성 함수
def generate_unique_name(category, index):
    return f"{category}_image_{index}.jpg"

# 폴더 생성 함수
def create_folder_structure(base_path):
    for split in ["train", "test"]:
        for category in categories.keys():
            path = os.path.join(base_path, split, category)
            os.makedirs(path, exist_ok=True)

# 이미지 파일을 무작위로 선택하고 복사하는 함수
def split_and_copy_images():
    create_folder_structure(destination_dir)

    # 학습/검증 데이터 개수 설정
    train_healthy_count = 500
    train_diseased_count = 500
    train_background_count = 200

    test_healthy_count = 100
    test_diseased_count = 200
    test_background_count = 50

    healthy_images = []
    diseased_images = []
    background_images = []

    # 각 폴더에서 이미지 분류
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        all_images = list(Path(folder_path).rglob("*.jpg"))
        if folder_name == "Background_without_leaves":
            background_images.extend(all_images)
        elif "healthy" in folder_name:
            healthy_images.extend(all_images)
        else:
            diseased_images.extend(all_images)

    # 중복 제거 및 섞기
    healthy_images = list(set(healthy_images))
    diseased_images = list(set(diseased_images))
    background_images = list(set(background_images))
    random.shuffle(healthy_images)
    random.shuffle(diseased_images)
    random.shuffle(background_images)

    # 학습/검증 데이터 나누기
    train_healthy = healthy_images[:train_healthy_count]
    test_healthy = healthy_images[train_healthy_count:train_healthy_count + test_healthy_count]

    train_diseased = diseased_images[:train_diseased_count]
    test_diseased = diseased_images[train_diseased_count:train_diseased_count + test_diseased_count]

    train_background = background_images[:train_background_count]
    test_background = background_images[train_background_count:train_background_count + test_background_count]

    # 학습용 이미지 복사
    for idx, image_path in enumerate(train_healthy):
        dest_path = os.path.join(destination_dir, "train", "healthy", generate_unique_name("healthy", idx))
        shutil.copy(image_path, dest_path)
        if (idx + 1) % 25 == 0:
            print(f"[TRAIN] Copied healthy image {idx + 1}/{train_healthy_count}")
        if idx+1 is train_healthy_count:
            print(f"=================================\n[TRAIN] healthy image 복사 완료\n=================================\n")


    for idx, image_path in enumerate(train_diseased):
        dest_path = os.path.join(destination_dir, "train", "diseased", generate_unique_name("diseased", idx))
        shutil.copy(image_path, dest_path)
        if (idx + 1) % 25 == 0:
            print(f"[TRAIN] Copied diseased image {idx + 1}/{train_diseased_count}")
        if idx+1 is train_diseased_count:
            print(f"================================\n[TRAIN] diseased image 복사 완료\n================================\n")

    for idx, image_path in enumerate(train_background):
        dest_path = os.path.join(destination_dir, "train", "background", generate_unique_name("background", idx))
        shutil.copy(image_path, dest_path)
        if (idx + 1) % 25 == 0:
            print(f"[TRAIN] Copied background image {idx + 1}/{train_background_count}")
        if idx+1 is train_background_count:
            print(f"================================\n[TRAIN] background image 복사 완료\n================================\n")

    # 검증용 이미지 복사
    for idx, image_path in enumerate(test_healthy):
        dest_path = os.path.join(destination_dir, "test", "healthy", generate_unique_name("healthy", idx))
        shutil.copy(image_path, dest_path)
        if (idx + 1) % 25 == 0:
            print(f"[TEST] Copied healthy image {idx + 1}/{test_healthy_count}")
        if idx+1 is test_healthy_count:
            print(f"\n================================\n[TEST] healthy image 복사 완료\n================================\n")

    for idx, image_path in enumerate(test_diseased):
        dest_path = os.path.join(destination_dir, "test", "diseased", generate_unique_name("diseased", idx))
        shutil.copy(image_path, dest_path)
        if (idx + 1) % 25 == 0:
            print(f"[TEST] Copied diseased image {idx + 1}/{test_diseased_count}")
        if idx+1 is test_diseased_count:
            print(f"\n================================\n[TEST] diseased image 복사 완료\n================================\n")

    for idx, image_path in enumerate(test_background):
        dest_path = os.path.join(destination_dir, "test", "background", generate_unique_name("background", idx))
        shutil.copy(image_path, dest_path)
        if (idx + 1) % 25 == 0:
            print(f"[TEST] Copied background image {idx + 1}/{test_background_count}")
        if idx+1 is test_background_count:
            print(f"\n================================\n[TEST] background image 복사 완료\n================================\n")

if __name__ == "__main__":
    split_and_copy_images()