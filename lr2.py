import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def f(x_):
    return (N + 1) / (N + 2) * np.cos(2 * N / (3 + N) * x_)


# определяем функцию для вычисления значения кубического сплайна
def S(x_):
    i = 0
    while i < Lenght - 1 and x_ > x[i + 1]:
        i += 1
    if i == Lenght - 1:
        i -= 1
    t = x_ - x[i]
    return y[i] + b[i] * t + c[i] * t ** 2 + d[i] * t ** 3


a = 0
b = 5
n = 10
h_ = (b-a)/n
x = [a + i * h_ for i in range(n)]
N = 6
y = [f(xi) for xi in x]

d = {'x': x, 'f(x)':y}
df = pd.DataFrame(data=d)
df.to_string(index=False)
print(df)
#x = [4.302, 4.381, 4.626, 4.886, 4.808, 4.872, 4.382, 4.181, 4.483, 4.418]
#y = [5.861, 6.212, 2.868, 2.647, 6.198, 3.499, 3.529, 6.511, 5.955, 4.185]

Lenght = len(x)

# Sort x and y arrays in ascending order of x values
sorted_indices = np.argsort(x)
x = [x[i] for i in sorted_indices]
y = [y[i] for i in sorted_indices]
N = 6

h = np.zeros(Lenght - 1)
alpha = np.zeros(Lenght - 1)
FirstInterpolCoefficient = np.zeros(Lenght)
SecondInterpolCoefficient = np.zeros(Lenght)
ThirdInterpolCoefficient = np.zeros(Lenght)
c = np.zeros(Lenght)
b = np.zeros(Lenght - 1)
d = np.zeros(Lenght - 1)

for i in range(Lenght - 1):
    h[i] = x[i + 1] - x[i]

for i in range(1, Lenght - 1):
    alpha[i] = (3 / h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (y[i] - y[i - 1])

# Метод прогонки
FirstInterpolCoefficient[0] = 1
SecondInterpolCoefficient[0] = 0
ThirdInterpolCoefficient[0] = 0

for i in range(1, Lenght - 1):
    FirstInterpolCoefficient[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * SecondInterpolCoefficient[i - 1]
    SecondInterpolCoefficient[i] = h[i] / FirstInterpolCoefficient[i]
    ThirdInterpolCoefficient[i] = (alpha[i] - h[i - 1] * ThirdInterpolCoefficient[i - 1]) / FirstInterpolCoefficient[i]

FirstInterpolCoefficient[Lenght - 1] = 1
ThirdInterpolCoefficient[Lenght - 1] = 0
c[Lenght - 1] = 0

# обратный ход
for j in range(Lenght - 2, -1, -1):
    c[j] = ThirdInterpolCoefficient[j] - SecondInterpolCoefficient[j] * c[j + 1]
    b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
    d[j] = (c[j + 1] - c[j]) / (3 * h[j])

print("Матрица:")

# создаем диагональную матрицу
diag = np.diag(np.round(FirstInterpolCoefficient, 3))

# создаем матрицы с смещением от главной диагонали
updiag = np.diagflat(np.round(ThirdInterpolCoefficient[:-1], 3), 1)
downdiag = np.diagflat(np.round(SecondInterpolCoefficient[1:], 3), -1)

# складываем все три матрицы
matrix = diag + updiag + downdiag

# создаем dataframe
df = pd.DataFrame(matrix)

# задаем имена столбцов и индексы строк
df.columns = [f'col{i + 1}' for i in range(df.shape[1])]
df.index = [f'row{i + 1}' for i in range(df.shape[0])]

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# выводим dataframe
print(df)

print("Коэффициенты кубического сплайна:")
for i in range(Lenght - 1):
    print(
        f"S{i + 1}(x) = {round(y[i], 3)} + {round(b[i], 3)}(x - {round(x[i], 3)}) + {round(c[i], 3)} (x - {round(x[i], 3)})^2 + {round(d[i], 3)} (x - {round(x[i], 3)})^3 = {S(x[i])}")

# задаем интервал для построения графика
xmin, xmax = min(x), max(x)
step = (xmax - xmin) / 1000
x_vals = np.arange(xmin, xmax + step, step)

# строим график точек и кубического сплайна
plt.scatter(x, y, color='red', label='Заданные точки')
plt.plot(x_vals, [S(x) for x in x_vals], label='Кубический сплайн')
x1 = np.linspace(min(x), max(x), 1000)
y1 = [f(xi) for xi in x1]
plt.plot(x1, y1, label='f(x)')
# добавляем легенду и подписи осей
plt.legend()
plt.xlabel('x')
plt.ylabel('y')

# выводим график
plt.show()
