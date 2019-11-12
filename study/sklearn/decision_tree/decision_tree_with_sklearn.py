
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
# 결정 트리가 많을수록 과대적합될 가능성이 있음
from sklearn.tree import DecisionTreeClassifier

# 결정 트리 만들기
tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
# plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105, 150))

# plt.xlabel('petal length [cm]')
# plt.ylabel('petal width [cm]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

dot_data = export_graphviz(tree, filled=True, rounded=True, 
                            class_names=['Setosa', 'Versicolor', 'Virginica'],
                            feature_names=['petal length', 'petal width'],
                            out_file=None )
# 결정 트리 를 잘그려줌
graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')

'''
랜덤 포레스트: 결정 트리의 앙상블
여러개의 깊은 결정 트리를 평균 내는것 -> 각각 트리는 분산이 높은 문제가 있지만 앙상블을 통해 견고한 모델을 만들어 일반화의 성능을 높임
1. n개의 랜덤한 부트스트랩 샘플을 뽑음(중복을 허용하면서 랜덤하게 n개의 샘플을 선택)
2. 부트스트랩 샘플에서 결정 트리를 학습(중복을 허용하지 않고 랜덤하게 d개의 특성을 선택)
3. 1,2번 단계를 k 번 반복
4. 각 트리의 예측을 모아 다수결 투표로 클래스 레이블을 할당

중복을 허용한 샘플링은ㄴ 샘플이 독립적이고 공분산이 0

보통 부트스트랩 샘플 크기를 원본 훈련 세트의 샘플 개수와 동일하게 함 -> 균형잡힌 편향-분산 트레이드 오프를 얻음
분할에 사용할 특성 개수 d는 훈련 세트에 있는 전체 특성 개수보다 작게 지정 => 전체 특성의 제곱근 값을 주로 사용 

'''
from sklearn.ensemble import RandomForestClassifier

# n_jobs : 컴퓨터의 멀티 코어를 사용해서 모델 훈련을 병렬화
forest = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()