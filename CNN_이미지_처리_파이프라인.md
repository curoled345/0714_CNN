
# 🧠 합성곱 신경망(CNN) 이미지 처리 파이프라인

이 문서는 **합성곱 신경망(CNN)**의 개념, 구조, 코드 및 악어 이미지를 활용한 실제 예제를 포함한 전체적인 내용을 설명합니다.

---

## 📌 목차

1. [CNN이란?](#-cnn이란)
2. [CNN의 구조](#-cnn의-구조)
3. [합성곱 연산](#-합성곱-연산)
4. [이미지 처리 파이프라인](#-이미지-처리-파이프라인)
5. [Keras를 이용한 CNN 구현](#-keras를-이용한-cnn-구현)
6. [특징 시각화](#-특징-시각화)
7. [CNN 활용 분야](#-cnn-활용-분야)
8. [디렉토리 구조](#-디렉토리-구조)
9. [필수 라이브러리](#-필수-라이브러리)
10. [참고 자료](#-참고-자료)

---

## 🧠 CNN이란?

**합성곱 신경망(CNN)**은 이미지와 같은 구조화된 데이터를 처리하는 데 뛰어난 성능을 보이는 딥러닝 알고리즘입니다. 계층적으로 이미지의 특징을 추출하여 학습합니다.

**특징:**

- 공간적 구조를 학습 가능
- 필터(커널) 공유로 학습 효율 향상
- 이미지 분류, 물체 인식 등에 매우 강력

---

## 🏗️ CNN의 구조

일반적인 CNN 구성은 다음과 같습니다:

1. **입력층**: 이미지 (예: 224x224x3)
2. **합성곱 층**: 특징 추출 필터 적용
3. **활성화 함수 (ReLU)**: 비선형성 부여
4. **풀링 층**: 특징 맵 축소 (예: MaxPooling)
5. **완전 연결층 (Dense)**: 분류 수행
6. **출력층**: 최종 예측 (Softmax 등)

```
입력 → [합성곱 → ReLU → 풀링] → ... → 전개 → Dense → 출력
```

---

## 🧮 합성곱 연산

합성곱 연산은 필터(커널)를 이미지 위에 슬라이딩하여 특징을 추출합니다.

**예시 필터 (엣지 검출):**
```
[[-1,  0,  1],
 [-2,  0,  2],
 [-1,  0,  1]]
```

```python
import cv2
import numpy as np

img = cv2.imread("images/article_1686275574.jpg", cv2.IMREAD_GRAYSCALE)
kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
edge = cv2.filter2D(img, -1, kernel)
cv2.imwrite("results/edge.jpg", edge)
```

---

## 🖼️ 이미지 처리 파이프라인

CNN에 이미지를 입력하기 전, 다음과 같은 전처리가 필요합니다:

1. 이미지 크기 변경 (224x224)
2. 픽셀 정규화 (0–1 범위로)
3. 배치 차원 추가

```python
def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)
```

---

## 🛠️ Keras를 이용한 CNN 구현

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

---

## 🧪 특징 시각화

CNN이 학습한 중간 특징 맵을 시각화할 수 있습니다:

```python
from tensorflow.keras.models import Model

layer_outputs = [layer.output for layer in model.layers[:4]]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

img = preprocess_image("images/article_1686275574.jpg")
activations = activation_model.predict(img)

import matplotlib.pyplot as plt
plt.imshow(activations[0][0, :, :, 0], cmap='viridis')
plt.title("첫 번째 필터 맵")
plt.colorbar()
plt.show()
```

---

## 📈 CNN 활용 분야

- 🖼️ 이미지 분류
- 🎯 객체 탐지
- 🧬 의료 영상 분석
- 🚗 자율 주행
- 🛰️ 위성 이미지 처리

---

## 📂 디렉토리 구조

```
cnn-image-demo/
├── images/
│   └── article_1686275574.jpg
├── results/
│   └── edge.jpg
├── src/
│   ├── preprocess.py
│   ├── model.py
│   └── visualize.py
└── README.md
```

---

## 📦 필수 라이브러리

```bash
pip install tensorflow opencv-python numpy matplotlib
```

---

## 📚 참고 자료

- [CS231n - 스탠포드 CNN 강의](http://cs231n.stanford.edu/)
- [DeepLearning.ai CNN 강의](https://www.deeplearning.ai/)
- [Keras 예제](https://keras.io/examples/vision/)
