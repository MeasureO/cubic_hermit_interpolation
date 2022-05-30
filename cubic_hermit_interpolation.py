import math
import matplotlib.pyplot as plt
import numpy as np
n = 4
def g(x):
    return x ** 10 - x ** 8
def f(x):
    a = [math.pi / n * i for i in range(n + 1)]
    y = [a[i] ** 10 - a[i] ** 8 for i in range(n + 1)]
    dy = [10 * a[i] ** 9 - 8 * a[i] ** 7 for i in range(n + 1)]
    res = []
    for i in range(0, n):
        if a[i] <= x <= a[i + 1]:
            b = np.array([y[i], y[i + 1], dy[i], dy[i + 1]])
            a = np.array([[a[i] ** (3 - j) for j in range(4)], [a[i + 1] ** (3 - j) for j in range(4)], [3 * a[i] ** 2, 2 * a[i], 1, 0], [3 * a[i + 1] ** 2, 2 * a[i + 1], 1, 0]])
            res = np.linalg.solve(a, b)
            # print(res)
            return res[0] * x ** 3 + res[1] * x ** 2 + res[2] * x  + res[3] 

x = [math.pi / n * i for i in range(n + 1)]
xx = [math.pi / 1000 * i for i in range(1000)]
y = [x[i] ** 10 - x[i] ** 8 for i in range(n + 1)]
dy = [10 * x[i] ** 9 - 8 * x[i] ** 7 for i in range(n + 1)]
res = []

plt.plot(xx, list(map(g, xx)), 'r')
plt.plot(xx, list(map(f, xx)), 'b')
plt.xlim(0, 5)
plt.ylim(-50, 50)
plt.savefig('test.png')



# for i in range(0, n):
#     b = np.array([y[i], y[i + 1], dy[i], dy[i + 1]])
#     a = np.array([[x[i] ** (3 - j) for j in range(4)], [x[i + 1] ** (3 - j) for j in range(4)], [3 * x[i] ** 2, 2 * x[i], 1, 0], [3 * x[i + 1] ** 2, 2 * x[i + 1], 1, 0]])
#     res.append(np.linalg.solve(a, b))


    
