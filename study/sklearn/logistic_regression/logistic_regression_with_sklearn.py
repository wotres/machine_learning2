# 로지스틱 회귀분석의 종속변수 값은 0, 1
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

# 붓꽃 데이터
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

# 특성 표준화
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

from sklearn.linear_model import LogisticRegression

# solver: 최적화에 사용할 알고리즘 / multi_class:'auto' 를 택하면 이진분류나 liblinear일 경우에는 ovr, 그외는 multinomizl을 선택 / C: 규칙강도의 역수값
lr = LogisticRegression(solver='liblinear', multi_class='auto', C=1000.0, random_state=1)
lr.fit(X_train_std, y_train)

# plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105,150))
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

# 3개의 샘플 테스트
# 첫번째 행은 첫번째 붓꽃의 클래스 소속 확률
print(lr.predict_proba(X_test_std[:3, :]))
# 모든 열을 더하면 1 -> 확률값의 합이므로
print(lr.predict_proba(X_test_std[:3, :]).sum(axis=1))
# 확률 예측한 결과 확인
print(lr.predict_proba(X_test_std[:3, :]).argmax(axis=1))
print(lr.predict(X_test_std[:3, :]))
# reshape의 -1 은 변경된 배열의 -1 위치의 차원은 원래 배열의 길이와 남은 차원으로 부터 추정된다는 뜻
# 하나의 행을 2차원 배열로 변경한느 법
print(lr.predict(X_test_std[0, :].reshape(1,-1)))

# 규제를 사용해 과대 적합 피하기

weights, params = [], []
for c in np.arange(-5, 5):
    # 다중분류에는 ovr 기법 사용 C 는 난다의 역수값, C 가 작아지면 가중치 절대값이 줄어듬. 즉, 규제 강도가 높아진다.
    lr = LogisticRegression(solver='liblinear', multi_class='auto', C=10.**c, random_state=1)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.**c)

weights = np.array(weights)
plt.plot(params, weights[:, 0], label='petal length')
plt.plot(params, weights[:, 1], linestyle='--', label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()
