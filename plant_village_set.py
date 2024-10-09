import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# 1. EfficientNet 모델 불러오기
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 2. 분류기 층 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # 이진 분류 (건강한 잎 vs 병든 잎)

# 3. 전체 모델 구성
model = Model(inputs=base_model.input, outputs=predictions)

# 4. 베이스 모델 가중치 고정 (전이 학습)
for layer in base_model.layers:
    layer.trainable = False

# 5. 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. 데이터 전처리 및 증강
train_datagen = ImageDataGenerator(
    rescale=1./255,         # 픽셀 값 정규화
    rotation_range=40,      # 이미지 회전
    width_shift_range=0.2,  # 가로 이동
    height_shift_range=0.2, # 세로 이동
    shear_range=0.2,        # 전단 변환
    zoom_range=0.2,         # 확대
    horizontal_flip=True,   # 수평 뒤집기
    fill_mode='nearest'     # 빈 부분 채우기
)

# 학습용 데이터 생성
train_generator = train_datagen.flow_from_directory(
    'path_to_plant_village_dataset/train',  # Plant Village 데이터 경로
    target_size=(224, 224),                 # EfficientNet 입력 크기
    batch_size=32,
    class_mode='binary'                     # 이진 분류
)

# 검증용 데이터 생성
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    'path_to_plant_village_dataset/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# 7. 모델 훈련
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=1
)

# 8. 전이 학습 (fine-tuning)을 위한 일부 레이어 풀기
for layer in base_model.layers[-20:]:
    layer.trainable = True

# 9. 모델 재컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# 10. 모델 재훈련
history_finetune = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=1
)
