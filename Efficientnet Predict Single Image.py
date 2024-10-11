import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

# 모델 불러오기
# model = tf.keras.models.load_model(r"C:\Users\jusan\Downloads\efficientnetB0_plant_classifier_epoch_20.h5")
model = tf.keras.models.load_model(r"C:\Users\jusan\Downloads\efficientnetB4_plant_classifier_epoch_10.h5")

# 이미지 경로 설정 (사용자가 입력한 경로 사용)
# image_path = r"C:\sg0117\model_test\graph_healthy_image (19).JPG"
image_path = r"C:\Users\jusan\OneDrive\바탕 화면\24-2캡스톤\자금우잎사진6.png"

# 이미지 전처리 함수
def preprocess_image(img_path):
    # B0은 타겟 사이즈 224로 해줘야함. B4는 380
    img = image.load_img(img_path, target_size=(380, 380))  # 모델 입력 크기에 맞게 이미지 로드 및 크기 조정
    img_array = image.img_to_array(img)  # 이미지를 배열로 변환
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    img_array = img_array / 255.0  # 픽셀 값을 0-1 범위로 스케일링
    return img_array

# 예측 함수
def predict_image(model, img_path):
    img = preprocess_image(img_path)  # 이미지 전처리
    predictions = model.predict(img)  # 모델 예측 수행
    class_indices = {0: 'Background', 1: 'Diseased', 2: 'Healthy'}  # 클래스 인덱스 매핑
    predicted_class = np.argmax(predictions[0])  # 가장 높은 확률을 가진 클래스 인덱스
    predicted_label = class_indices[predicted_class]  # 클래스 라벨

    if predicted_label == 'Background':
        print("잎 사진이 아닙니다.")
    else:
        print(f"이 잎은 {predicted_label}입니다.")

# 예측 실행
predict_image(model, image_path)
