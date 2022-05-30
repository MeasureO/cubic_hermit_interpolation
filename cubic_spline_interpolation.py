import math
import numpy as np
import matplotlib.pyplot as plt
import pprint 

n = 8


x = [10 / n * i for i in range(n + 1)]
xx = [10 / 1000 * i for i in range(1000)]
y = [math.cos(x[i]) * x[i] ** 2 - x[i] * math.sin(x[i]) for i in range(n + 1)]
# dy = [10 * x[i] ** 9 - 8 * x[i] ** 7 for i in range(n + 1)]

d = [[0 for i in range(n + 1)] for j in range(n + 1)]
b = [0] * (n + 1)
for i in range(1, n):
    d[i][i - 1] = x[i + 1] - x[i]
    d[i][i] = 2 * (x[i + 1] - x[i - 1])
    d[i][i + 1] = x[i] - x[i - 1]
    b[i] = 3 * (y[i] - y[i - 1]) * (x[i + 1] - x[i]) / (x[i] - x[i - 1]) + 3 * (y[i + 1] - y[i]) * (x[i] - x[i - 1]) / (x[i + 1] - x[i])
e1 = -math.sin(x[0]) * x[0] ** 2 + 2 * x[0] * math.cos(x[0]) - math.sin(x[0]) - x[0] * math.cos(x[0])
e2 = -math.sin(x[n]) * x[n] ** 2 + 2 * x[n] * math.cos(x[n]) - math.sin(x[n]) - x[n] * math.cos(x[n])
d[0] = [1] + [0] * n
d[n] = [0] * n + [1]
b[0] = e1
b[n] = e2
print(x)
pprint.pprint(d)
pprint.pprint(b)
np_d = np.array(d)
np_b = np.array(b)
dy = np.linalg.solve(np_d, np_b)
pprint.pprint(dy)

def g(x):
    return math.cos(x) * x ** 2 - x * math.sin(x)

def f(x):
    a = [10 / n * i for i in range(n + 1)]
    y = [math.cos(a[i]) * a[i] ** 2 - a[i] * math.sin(a[i]) for i in range(n + 1)]
    res = []
    for i in range(0, n):
        if a[i] <= x <= a[i + 1]:
            b = np.array([y[i], y[i + 1], dy[i], dy[i + 1]])
            a = np.array([[a[i] ** (3 - j) for j in range(4)], [a[i + 1] ** (3 - j) for j in range(4)], [3 * a[i] ** 2, 2 * a[i], 1, 0], [3 * a[i + 1] ** 2, 2 * a[i + 1], 1, 0]])
            res = np.linalg.solve(a, b)
            # print(res)
            return res[0] * x ** 3 + res[1] * x ** 2 + res[2] * x  + res[3] 


plt.plot(xx, list(map(g, xx)), 'r')
plt.plot(xx, list(map(f, xx)), 'b')
plt.xlim(0, 10)
plt.ylim(-50, 50)
plt.savefig('test.png')
