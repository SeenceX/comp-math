import numpy as np
import matplotlib.pyplot as plt

X, T = 1, 1
m, n = 10, 250
N = 6
h = X / m
t = T / n

p = lambda N: (3 * N + 1) / (N + 2)

U1 = np.zeros((m + 1, n + 1))
U2 = np.zeros((m + 1, 2))

for i in range(m + 1):
    for j in range(n + 1):
        if j == 0 and 0 < i < m:
            U1[i][j] = p(N) * np.sin(np.pi * (i * h))
        else:
            U1[i][j] = 0

for i in range(1, m):
    for j in range(1, n):
        U1[i][j] = ((U1[i - 1][j - 1] - 2.0 * U1[i][j - 1] + U1[i + 1][j - 1]) * t / (h * h) + U1[i][j - 1])
        if j * t == 0.2:
            U2[i][0] = U1[i][j]
            U2[i][1] = i * h

x = U2[:len(U2) - 1, 1]
y = U2[:len(U2) - 1, 0]
x = np.append(x, 1)
y = np.append(y, 0)

for i in range(int(m)):
    print(f"({x[i]}; {y[i]})")

# Строим график
plt.plot(x, y, '-*')
plt.xlabel('x')
plt.ylabel('U(x, t=0.2)')
plt.title('График функции U(x, t=0.2)')
plt.show()
