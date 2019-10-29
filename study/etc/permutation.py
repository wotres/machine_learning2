import numpy as np
# np.random.permutation(len(x)) : 해당 배열 x 의 길이만큼 순서를 랜덤으로 내어준다.
temp = ['a','b','c','d']
print(np.random.permutation(len(temp)))
print(np.random.permutation(temp))
print(temp)
# np.random.shuffle(x) : 해당 배열 x 를 섞어준다.
x = np.arange(10)
print(x)
np.random.shuffle(x)
print(x)
