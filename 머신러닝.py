from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # 추가된 부분
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # 추가된 부분

def generate_safety_dataset(num_samples=1000):
    np.random.seed(42)
    # 화학물질 노출량 (가상의 데이터)
    chemical_exposure = np.random.uniform(0, 100, num_samples)
    # 안전 장비 사용 여부 (가상의 데이터, 0: 미사용, 1: 사용)
    safety_equipment_used = np.random.choice([0, 1], num_samples)
    # 안전사고 여부 (가상의 데이터, 0: 발생하지 않음, 1: 발생)
    safety_incident = np.random.choice([0, 1], num_samples, p=[0.95, 0.05])

    # 데이터프레임 생성
    safety_data = pd.DataFrame({
        'Chemical_Exposure': chemical_exposure,
        'Safety_Equipment_Used': safety_equipment_used,
        'Safety_Incident': safety_incident
    })

    return safety_data

# 데이터셋 생성
safety_dataset = generate_safety_dataset()
# 데이터셋 준비
X = safety_dataset[['Chemical_Exposure', 'Safety_Equipment_Used']]
y = safety_dataset['Safety_Incident']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 머신러닝 모델 선택 및 훈련
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# 훈련된 모델을 사용한 예측
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# 모델 평가
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Test Accuracy: {test_accuracy:.2f}')

# 분류 보고서 및 혼동 행렬 출력
print('\nClassification Report:')
print(classification_report(y_test, test_predictions, zero_division=1))  # zero_division 매개변수 추가

print('\nConfusion Matrix:')
print(confusion_matrix(y_test, test_predictions))

import matplotlib.pyplot as plt

# 훈련 데이터의 예측 결과 시각화
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_train['Chemical_Exposure'], X_train['Safety_Equipment_Used'], c=train_predictions, cmap='coolwarm', edgecolors='k', marker='o', alpha=0.8)
plt.title('Training Predictions')
plt.xlabel('Chemical Exposure')
plt.ylabel('Safety Equipment Used')
# 테스트 데이터의 예측 결과 시각화
plt.subplot(1, 2, 2)
plt.scatter(X_test['Chemical_Exposure'], X_test['Safety_Equipment_Used'], c=test_predictions, cmap='coolwarm', edgecolors='k', marker='o', alpha=0.8)
plt.title('Test Predictions')
plt.xlabel('Chemical Exposure')
plt.ylabel('Safety Equipment Used')

plt.tight_layout()
plt.show()

