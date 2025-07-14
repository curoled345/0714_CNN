# 🧠 Convolutional Neural Network(CNN) 용어 정리

## 📘 기본 용어

### 1. Convolution (합성곱)
- 입력 이미지에 필터(커널)를 적용하여 특징 맵(feature map)을 생성하는 연산
- 특징 추출의 핵심 역할

### 2. Kernel / Filter (커널 / 필터)
- 작은 크기의 행렬로, 입력 이미지에서 특징을 추출하는 데 사용됨
- 일반적으로 3x3, 5x5 크기를 가짐

### 3. Feature Map (특징 맵)
- 커널이 입력 데이터에 적용된 결과
- 입력 이미지의 특정 패턴이나 특성을 강조한 출력

### 4. Stride (스트라이드)
- 커널이 입력 이미지를 가로질러 이동하는 간격
- 스트라이드가 클수록 출력 크기는 작아짐

### 5. Padding (패딩)
- 입력 가장자리에 추가되는 값(주로 0)을 의미
- 출력 크기를 조절하거나 경계 정보 보존에 사용

- **Valid Padding**: 패딩 없이 연산. 출력 크기 작아짐
- **Same Padding**: 출력 크기를 입력과 동일하게 유지하도록 패딩 추가

---

## 🧱 신경망 구조 구성 요소

### 6. Input Layer (입력층)
- 원본 이미지가 모델에 처음 입력되는 층

### 7. Convolutional Layer (합성곱 층)
- 필터를 이용해 입력 이미지에서 특징을 추출하는 층

### 8. Activation Function (활성화 함수)
- 비선형성을 추가하여 모델이 복잡한 함수도 학습할 수 있도록 도와줌
- 대표적으로 **ReLU (Rectified Linear Unit)** 사용됨

### 9. Pooling Layer (풀링 층)
- 공간 정보를 축소하여 계산량과 과적합을 줄임
- **Max Pooling**: 윈도우 내 최대값 선택
- **Average Pooling**: 윈도우 내 평균값 선택

### 10. Flatten (플래튼)
- 다차원 데이터를 1차원으로 펼치는 작업
- Fully Connected Layer에 전달하기 전 수행됨

### 11. Fully Connected Layer (완전 연결 층)
- 모든 뉴런이 이전 층의 모든 뉴런과 연결
- 최종 분류를 수행함

### 12. Output Layer (출력층)
- 최종 결과를 출력
- 다중 클래스 분류에서는 **Softmax** 함수 사용

---

## 🛠 학습 관련 용어

### 13. Loss Function (손실 함수)
- 모델 예측과 실제 값의 차이를 측정
- 대표적으로 **Cross-Entropy Loss**, **MSE** 등이 있음

### 14. Backpropagation (역전파)
- 오차를 기반으로 가중치를 조정하는 알고리즘

### 15. Optimization Algorithm (최적화 알고리즘)
- 손실 함수 값을 최소화하도록 파라미터를 조정
- 대표적으로 **SGD**, **Adam**, **RMSProp** 등이 있음

### 16. Epoch (에폭)
- 전체 학습 데이터를 한 번 모두 학습시키는 과정

### 17. Batch Size (배치 크기)
- 한 번에 학습에 사용되는 샘플 수

### 18. Overfitting (과적합)
- 학습 데이터에는 잘 맞지만, 새로운 데이터에는 일반화 성능이 떨어지는 현상

---

## 📊 시각화 및 분석

### 19. Feature Visualization
- 중간 층의 feature map을 시각화하여 모델이 어떤 특징을 학습했는지 확인

### 20. Grad-CAM
- CNN이 특정 예측을 할 때 어떤 영역에 주목했는지를 시각적으로 표현하는 기법

---

## 📚 참고
- DeepLearning.ai, CS231n, PyTorch Docs, TensorFlow Tutorials

