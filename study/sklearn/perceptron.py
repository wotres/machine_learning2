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

from sklearn.metrics import accuracy_score
print('정확도: %.2f' % accuracy_score(y_test, y_pred))
print('정확도: %.2f' % ppn.score(X_test_std, y_test))

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 결정 경계
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() +1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                        np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    print(X)
    for idx, cl in enumerate(np.unique(y)):
        print(X[y==cl, 0])
        # y==cl 로 비교하여 하나씩 그린다. 
        # [true, false] 와 같은 배열로 true 인 곳만 골라서 그린다.
        plt.scatter(x=X[y==cl, 0],
                        y=X[y==cl,1],
                        alpha=0.8,
                        c=colors[idx],
                        marker=markers[idx],
                        label=cl,
                        edgecolor='black')

    print(test_idx)
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        print(y_test)
        # s: size, marker : 모양 (s 는 스퀘어), alpha: 투명도(1이면 완전 불투명)
        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set'
                    )

# print(X_train_std)
# np.vstack : 열수가 같은 두개 이상의 배열을 행으로 연결
X_combined_std = np.vstack((X_train_std, X_test_std))
# np.hstack : 행수가 같은 두개 이상의 배열을 열로 연결
y_combined = np.hstack((y_train, y_test))
# print(X_combined_std)

plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))

plt.xlabel('petal length [standardize]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
# 플롯간 간격 자동 맞춤
plt.tight_layout()
plt.show()


