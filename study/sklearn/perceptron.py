from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

print("클레스 레이블: ", np.unique(y))

# 7:3 으로 훈련 데이터: 테스트 데이터로 분할
from sklearn.model_selection import train_test_split
# train_test_split 는 분할전 데이터셋을 미리 섞음
# stratify=y 로 설정한다면 실제 정답셋의 비율이 1:2 이면 나눠진 데이터 셋들도 1:2의 비율을 유지한채 분할됨
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# np.bincount(x) : 각각 빈도 카운트
print('y 레이블 카운트:', np.bincount(y))
print('y 레이블 카운트:', np.bincount(y_train))
print('y 레이블 카운트:', np.bincount(y_test))

# 특성 표준화
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
# fit 으로 데이터 변환 학습 후 transform 으로 스케일 조정
# fit으로 각 특성 차원마다 샘플과 표준편차 계산
# transform 으로 표준화

sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import Perceptron
# eta0 : 학습률, tol : 어느 수준까지 진행할것인가를 결정(이전 loss 값과의 차이가 tol 보다 작으면 멈춤)
ppn = Perceptron(max_iter=40, eta0=0.1, tol=1e-3, random_state=1)
ppn.fit(X_train_std, y_train)
# 정확도 확인
y_pred = ppn.predict(X_test_std)
print('잘못 분류된 샘플 개수: %d' % (y_test !=y_pred).sum())



