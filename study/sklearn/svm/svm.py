'''
svm은 마진을 최대화 하는ㄴ 것
마진은 초평면(결정 경계)과 이 초평면에 가장 가까운 훈련 샘플 사이의 거리로 정의
마진을 최대할 할수록 일반화 오차가 낮아지는 경향이 있기 떄문

슬랙변수(소프트 마진 분류)
슬랙 변수는 선형적으로 구분되지 않는 데이터에서 선형 제약 조건을 완화할 필요가 있기 떄문에 도입
-> 이를 통해 적절히 비용을 손해 보면서 분류 오차가 있는 상황에서 최적화 알고리즘이 수렴

로지스틱이 서포트 벡터 머신보다 이상치에 민감
'''
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # 마커와 컬러맵을 설정합니다.
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 결정 경계를 그립니다.
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # 테스트 샘플을 부각하여 그립니다.
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')

from sklearn.svm import SVC

# C 값이 크면 오차에 대한 비용이 커짐, C 값이 작으면 분류 오차에 덜 엄격해짐
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)

# plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
# plt.scatter(svm.dual_coef_[0,:], svm.dual_coef_[1, :])
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')

# plt.tight_layout()
# plt.show()

# weight 
print(svm.coef_)
# 각 원소가 ai , yi 로 이루어진 벡터
print(svm.dual_coef_)
print(svm.dual_coef_.shape)

from sklearn.linear_model import SGDClassifier

ppn = SGDClassifier(loss='perceptron')
lr = SGDClassifier(loss='log')
svm = SGDClassifier(loss='hinge')

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)
# randn : 가우시안 표준 정규 분포 
X_xor = np.random.randn(200,2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)

y_xor = np.where(y_xor, 1, -1)

# plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor ==1, 1], c='b', marker='x', label='1')
# plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='r', marker='s', label='-1')

# plt.xlim([-3, 3])
# plt.ylim([-3, 3])
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()

'''
선형적으로 구분되지않는 데이터를 위해 커널 방법을 사용
매핑 함수 파이를 사용하여 원본 특성의 비선형 조합을 선형적으로 구분되는 고차원 공간에 투영
두 포인트 사이 점곱을 계산하는데 드는 비용을 절감하기 위해 커널함수를 정의
가우시안 커널을 주로 사용 
간단히 커널이라는 용어를 샘플 간의 유사도 함수로 해석할 수 있음
지수함수로 얻게 되는 유사도 점수는 1(매우 비슷함) 에서 0(매우 다름) 사이의 범위를 가짐
'''

# 커널 기법을 사용해 고차원 공간에서 분할 초평면 찾기
# gamma는 가우시안구의 크기를 제한하는 매개변수 
# -> r 값을 크게하면 서포트 벡터의 영향이나 범위가 줄어듬, 결정 경계는 더욱 샘플에 가까워지고 구불구불해짐
svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
# plot_decision_regions(X_xor, y_xor, classifier=svm)

# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)
# plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
# plt.scatter(svm.dual_coef_[0,:], svm.dual_coef_[1,:])
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

svm = SVC(kernel='rbf', random_state=1, gamma=100, C=1.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, 
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
