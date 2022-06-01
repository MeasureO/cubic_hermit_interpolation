import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def f(x, y):
   return x ** 2 + y ** 2

def make_F_ij(i, j, x, y, dzdx, dzdy):
   F_ij = [[f(x[i], y[j])    , dzdy[j], f(x[i], y[j + 1])    , dzdy[j + 1]], 
           [dzdx[i]          , 0      , dzdx[i]              , 0],
           [f(x[i + 1], y[j]), dzdy[j], f(x[i + 1], y[j + 1]), dzdy[j + 1]],
           [dzdx[i + 1]      , 0      , dzdx[i + 1]          , 0]]
   Fij = np.array(F_ij)
   return Fij

def make_A_inv(h):
   A_inv = [[1, 0, 0, 0], [0, 1, 0, 0], [-3 / h ** 3, -2 / h ** 2, 3 / h ** 3,  -1 / h ** 2], [2 / h ** 3 , 1 / h ** 2, -2 / h ** 3, 1 / h ** 2]]
   A_inv = np.array(A_inv)
   return A_inv

def make_G_ij(A_i, A_j, F_ij):
   return A_i @ F_ij @ np.transpose(A_j)

def make_approximation(xt, yt, x, y, i, j, dzdx, dzdy):
   Fij = make_F_ij(i, j, x, y, dzdx, dzdy)
   Ai = make_A_inv(x[i + 1] - x[i]) 
   Aj = make_A_inv(x[j + 1] - x[j])
   Gij = make_G_ij(Ai, Aj, Fij)
   value = 0
   for k in range(0, 4):
      for l in range(0, 4):
         value += Gij[k][l] * (xt - x[i]) ** k * (yt - y[j]) ** l
   return value

         

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

step = 0.2
x = np.arange(-1, 1.1, step)
y = np.arange(-1, 1.1, step)
xx = np.arange(-1, 1.01, 0.001)
yy = np.arange(-1, 1.01, 0.001)
X, Y = np.meshgrid(x, y)
XX, YY = np.meshgrid(xx, yy)
ZZ = XX ** 2 + YY ** 2
dzdx = 2 * x
dzdy = 2 * y 
dzdxdy = [0] * len(x)

# # Z = Y * np.sin(X) + np.sin(Y)

Z = np.zeros((len(x), len(y)))

for j in range(len(x)):
   for i in range(len(x)):
      if i != len(x) - 1 and j != len(y) - 1:
         Z[i][j] = make_approximation(x[i] + step / 2, y[j] + step / 2, x, y, i, j, dzdx, dzdy)
Z = np.array(Z)
Z[-1] = Z[0]
Z[:,-1] = Z[:,0]
print(Z)


surf = ax.plot_surface(XX, YY, ZZ, 
                       linewidth=0, antialiased=False)
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(0, 10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
