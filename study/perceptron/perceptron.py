## 퍼셉트론으로 붓꽃 분류
import numpy as numpy
class Perceptron(object):
    # eta: 학습률 / n_iter: 학습 반복 횟수 / random_state : 가중치 초기화를 위한 난수 시드
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    # w_[1:] 가중치, w_[0] 절편
    def fit(self, X, y):
        # 일단 난수 시드에 값생성
        rgen = np.random.RandomState(self.random_state)
        # 정규분포 평균/ 표준편차 / 뽑아낼 행,열 크기
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size= 1 + X.shape[1])
        self.errors_ = []
        print(self.w_)
        for _ in range(self.n_iter):
            errors = 0
            # 배치가 아니라 각각 값당 갱신이 일어남
            for xi, target in zip(X, y):
                diff = target - self.predict(xi)
                if(diff != 0 ):
                    print(diff)
                update = self.eta * diff
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(df.tail())
###############################################################
## 붓꽃 분포 확인 
import matplotlib.pyplot as plt
import numpy as np

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0, 2]].values
# print(X)

# plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

# plt.xlabel('sepal length [cm]')
# plt.ylabel('sepla length [cm]')
# plt.legend(loc='upper left')
# plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Number of errors')
# plt.show()

###############################################################
# 결정 경계 그래프
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # 마커와 컬러맵 설정
    markers = ('s','x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    # ListedColormap : 인자로 주어진 색상을 그래프상에 표시하기 위한 객체
    # np.unique(y) : y에 있는 고유한 값을 작은 값 순으로 나열 -> 같은 값 중복있는경우 하나만
    # 따라서 고유값이 -1, 1 이 나온 경우 각각 (s,red), (x, blue) 가 매핑됨.
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 결정 경계
    x1_min, x1_max = X[:, 0].min() -1, X[:, 0].max() +1
    x2_min, x2_max = X[:, 1].min() -1, X[:, 1].max() +1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))

    # ravel() : reshape() 와 반대로 하나로 풀어줌
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    # contourf 를 통해 등고선을 그림 -> 같은 높이에 있는 것들끼리 묶어줌
    # 가로축 값, 세로축값, 예측치(구분값) 즉 등고선 높이, 투명도, cmap(구분값에 따른 색상)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    # xlim : 그림 범위 수동 지정
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    print(y)

    # 샘플의 산점도
    for idx, cl in enumerate(np.unique(y)):
        print(idx, X[y==cl,0])
        plt.scatter(x=X[y==cl, 0],
                    y=X[y==cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()



