import tensorflow as tf  # TensorFlow 라이브러리 불러오기
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # 이미지 데이터 증강을 위한 모듈
from tensorflow.keras.applications import EfficientNetB0  # EfficientNetB0 모델 불러오기
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense  # 레이어 구성에 필요한 모듈
from tensorflow.keras.models import Sequential  # 순차 모델을 만들기 위한 모듈

# 데이터셋 경로 설정
dataset_dir = "C:/sg0117/dataset/data"  # 학습 및 테스트 데이터가 있는 디렉토리 경로

# 데이터 증강 설정 및 데이터셋 불러오기
train_datagen = ImageDataGenerator(rescale=1.0/255.0)  # 학습 데이터의 픽셀 값을 0-1 범위로 스케일링

test_datagen = ImageDataGenerator(rescale=1.0/255.0)  # 테스트 데이터의 픽셀 값을 0-1 범위로 스케일링

train_generator = train_datagen.flow_from_directory(  # 학습 데이터 생성기 설정
    f"{dataset_dir}/train",  # 학습 데이터 경로
    target_size=(224, 224),  # EfficientNet 입력 크기에 맞게 이미지 크기 조정
    batch_size=32,  # 배치 크기 설정
    class_mode='categorical'  # 세 가지 클래스 (healthy, diseased, background)를 위한 범주형 라벨 설정
)

test_generator = test_datagen.flow_from_directory(  # 테스트 데이터 생성기 설정
    f"{dataset_dir}/test",  # 테스트 데이터 경로
    target_size=(224, 224),  # 이미지 크기 조정
    batch_size=32,  # 배치 크기 설정
    class_mode='categorical'  # 범주형 라벨 설정
)

# EfficientNet 모델 구성
efficientnet_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  # 사전 학습된 EfficientNetB0 모델 불러오기, 최종 분류 레이어 제외
model = Sequential([  # 순차 모델 구성
    efficientnet_base,  # EfficientNetB0 기반 모델 추가
    GlobalAveragePooling2D(),  # 글로벌 평균 풀링 레이어 추가
    Dense(3, activation='softmax')  # 세 가지 클래스 분류 (healthy, diseased, background)를 위한 출력 레이어 추가
])

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # 모델 학습을 위한 컴파일 (손실 함수, 최적화 알고리즘, 평가지표 설정)

# 모델 학습
model.fit(  # 모델 학습 시작
    train_generator,  # 학습 데이터 제너레이터
    validation_data=test_generator,  # 검증 데이터 제너레이터
    epochs=10,  # 총 학습 반복 횟수 설정
    steps_per_epoch=len(train_generator),  # 한 epoch에서 학습할 스텝 수 설정
    validation_steps=len(test_generator)  # 한 epoch에서 검증할 스텝 수 설정
)

# 모델 저장
model.save("C:/sg0117/models/efficientnet_plant_classifier.h5")  # 학습된 모델을 지정된 경로에 저장