import numpy as np
import matplotlib.pyplot as plt
'''
np.arange()
범위 구간의 순차적 배열 형태로 반환해주는 함수
np.arange(시작, 끝, 증가값) -> 범위는 이상/미만임 , 디폴트 시작값은 0
'''
# print(np.arange(1,3,2))

###############################################################
'''
np.meshgrid(x,y)
x, y 각각의 배열이 존재할떄 이들의 배열 크기를 맞춰 준다. -> 크기를 맞춰 그림그리기 쉽게 해줌
'''
# x = np.arange(3)
# y = np.arange(4)

# xx, yy = np.meshgrid(x, y)

# print(xx)
# print(yy)

# dots = [list(zip(x,y)) for x, y in zip(xx, yy)]
# print(dots)

# plt.title("np.meshgrid")
# plt.scatter(xx, yy, linewidths=10)
# plt.show()

###############################################################
'''
np.reshape() : 원하는 행개수 열개수를 가진 배열의 형태를 만들어줌
np.ravel() : 배열의 형태를 풀어 1차원으로 나열해줌
'''
a = np.arange(12)
b = a.reshape(3,4)
print(b)

c = b.ravel()
print(c)

d = np.ravel(b)
print(d)