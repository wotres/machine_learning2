'''
결정 트리는 Information Gain 이 최대화 되도록 분류
불순도를 낮춰 차이가 크도록한다.
불순도 분할 조건은 지니불순도, 엔트로피, 분류 오차를 사용
모든 조건이 같은 확률일떄 (분류어려움으로) 불순도는 최대가 된다.
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



import matplotlib.pyplot as plt 
import numpy as np 

def gini(p):
    return p * (1-p) + (1-p) * (1 - (1- p))

def entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2((1-p))

def error(p):
    return 1- np.max([p, 1-p])

x = np.arange(0.0, 1.0, 0.01)

ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e * 0.5 if e else None for e in ent]
err = [error(i) for i in x]

# fig = plt.figure()
# # 1*1 그리더에 첫번째 subplot
# ax = plt.subplot(111)
# for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
#                             ['Entropy', 'Entropy (scaled)', 'Gini Impurity', 'Misclassification Error'],
#                             ['-','-','-','-'],
#                             ['black', 'lightgray', 'red', 'green','cyan']):
#     line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

# # bbox_to_anchor : 범례의 위치 조정 (좌우, 우아래)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, fancybox=True, shadow=False)

# ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
# ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
# plt.ylim([0, 1.1])
# plt.xlabel('p(i=1)')
# plt.ylabel('Impurity Index')
# plt.show()


