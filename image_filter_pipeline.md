
# 🐊 이미지 필터링 파이프라인 - 악어 이미지 예제

본 프로젝트는 `article_1686275574.jpg` 이미지를 이용하여 **수직 엣지 감지**, **수평 엣지 감지**, **블러**, **샤프닝** 등의 전처리 필터를 적용한 결과를 단계별로 보여줍니다.

---

## 📥 입력 이미지

![악어 이미지](images/article_1686275574.jpg)

---

## 📐 1. 수직 엣지 감지 (Vertical Edge Detection)

- 필터(kernel):

```
[[-1,  0,  1],
 [-2,  0,  2],
 [-1,  0,  1]]
```

- 설명: X축 방향(좌우)의 경계(엣지)를 강조

```python
import cv2
import numpy as np

img = cv2.imread("images/article_1686275574.jpg", cv2.IMREAD_GRAYSCALE)
kernel_vertical = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

vertical_edge = cv2.filter2D(img, -1, kernel_vertical)
cv2.imwrite("results/vertical_edge.jpg", vertical_edge)
```

> ![수직 엣지](results/vertical_edge.jpg)

---

## 📏 2. 수평 엣지 감지 (Horizontal Edge Detection)

- 필터(kernel):

```
[[-1, -2, -1],
 [ 0,  0,  0],
 [ 1,  2,  1]]
```

- 설명: Y축 방향(상하)의 경계(엣지)를 강조

```python
kernel_horizontal = np.array([[-1, -2, -1],
                              [ 0,  0,  0],
                              [ 1,  2,  1]])

horizontal_edge = cv2.filter2D(img, -1, kernel_horizontal)
cv2.imwrite("results/horizontal_edge.jpg", horizontal_edge)
```

> ![수평 엣지](results/horizontal_edge.jpg)

---

## 💧 3. 블러 처리 (Gaussian Blur)

- 필터: Gaussian Blur (5x5)
- 설명: 노이즈 제거, 윤곽 흐리기

```python
blurred = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imwrite("results/blurred.jpg", blurred)
```

> ![블러 이미지](results/blurred.jpg)

---

## ✨ 4. 샤프닝 (Sharpening)

- 필터(kernel):

```
[[ 0, -1,  0],
 [-1,  5, -1],
 [ 0, -1,  0]]
```

- 설명: 경계 강조, 이미지 선명도 향상

```python
kernel_sharpen = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]])

sharpened = cv2.filter2D(img, -1, kernel_sharpen)
cv2.imwrite("results/sharpened.jpg", sharpened)
```

> ![샤프닝 이미지](results/sharpened.jpg)

---

## 🧾 5. 최종 결과 비교

| 단계 | 이미지 |
|------|--------|
| 원본 | ![원본](images/article_1686275574.jpg) |
| 수직 엣지 | ![수직엣지](results/vertical_edge.jpg) |
| 수평 엣지 | ![수평엣지](results/horizontal_edge.jpg) |
| 블러 | ![블러](results/blurred.jpg) |
| 샤프닝 | ![샤프닝](results/sharpened.jpg) |

---

## 🛠️ 필수 라이브러리

```bash
pip install opencv-python numpy
```

---

## 📂 디렉토리 구조

```
image-filter-pipeline/
├── images/
│   └── article_1686275574.jpg
├── results/
│   ├── vertical_edge.jpg
│   ├── horizontal_edge.jpg
│   ├── blurred.jpg
│   └── sharpened.jpg
├── process_filters.py
└── README.md
```

---

## 📌 참고

- 모든 필터는 `cv2.filter2D()`로 적용
- 이미지 전처리는 CNN 학습 전 필수 과정
- 다양한 커널을 실험해보면 더 풍부한 결과 가능
