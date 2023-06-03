import matplotlib.pyplot as plt
import pandas as pd


class Gauss:
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def GaussMethod(self):
        n = len(self.A)
        M = self.A.copy()
        V = self.B.copy()

        for i in range(n):
            max_row = i
            for j in range(i + 1, n):
                if abs(M[j][i]) > abs(M[max_row][i]):
                    max_row = j
            M[i], M[max_row] = M[max_row], M[i]
            V[i], V[max_row] = V[max_row], V[i]

            if M[i][i] == 0:
                return None

            for j in range(i + 1, n):
                ratio = M[j][i] / M[i][i]
                V[j] -= ratio * V[i]
                for k in range(i, n):
                    M[j][k] -= ratio * M[i][k]

        X = [0 for i in range(n)]
        for i in range(n - 1, -1, -1):
            X[i] = V[i]
            for j in range(i + 1, n):
                X[i] -= M[i][j] * X[j]
            X[i] /= M[i][i]
        return X


def linear_dependence(x, y):
    n = len(x)
    x2_sum = sum(x[i] ** 2 for i in range(n))
    x_sum = sum(x)
    xy_sum = sum(x[i] * y[i] for i in range(n))
    y_sum = sum(y)
    A = [[x2_sum, x_sum], [x_sum, n]]
    B = [xy_sum, y_sum]
    return A, B


def quadratic_dependence(x, y):
    n = len(x)
    x_sum = sum(x)
    y_sum = sum(y)
    x2_sum = sum(x[i] ** 2 for i in range(n))
    xy_sum = sum(x[i] * y[i] for i in range(n))
    x2y_sum = sum(x[i] ** 2 * y[i] for i in range(n))
    x3_sum = sum(x[i] ** 3 for i in range(n))
    x4_sum = sum(x[i] ** 4 for i in range(n))
    A = [[x2_sum, x_sum, n], [x3_sum, x2y_sum, x_sum], [x4_sum, x3_sum, x2_sum]]
    B = [y_sum, xy_sum, x2y_sum]
    return A, B


f1 = lambda x, c: [c[0] * xi + c[1] for xi in x]  # линейная функция
f2 = lambda x, c: [c[0] * xi ** 2 + c[1] * xi + c[2] for xi in x]  # квадратичная функция


def graph(_x, _y, c1, c2):
    k = (max(_x) - min(_x)) * 0.01  # шаг
    x = [min(_x) + k * i for i in range(101)]  # набор из 100 точек в интервале от min до max значения данных X
    plt.scatter(_x, _y, color='blue')  # вывод точек
    plt.plot(x, f1(x, c1), color='red')  # вывод линейной функции
    plt.plot(x, f2(x, c2), color='green')  # вывод квадратичной функции

    plt.title('Метод наименьших квадратов')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()


# Вычисляем суммы квадратов отклонений исходных данных от линий и сравниваем их.
# Меньшее значение соответствует линии, которая лучше в смысле метода наименьших квадратов
# аппроксимирует исходные данные.
def inaccuracy(_x, _y, c1, c2):
    inac_1 = sum([(yi - (xi * c1[0] + c1[1])) ** 2 for xi, yi in zip(_x, _y)])
    inac_2 = sum([(yi - (xi ** 2 * c2[0] + c2[1] * xi + c2[2])) ** 2 for xi, yi in zip(_x, _y)])
    print()
    print(f"погрешность от f1(x) = {inac_1} | погрешность от f2(x) = {inac_2}")
    if inac_1 > inac_2:
        print("Так как погрешность f2(x) < f1(x), то прямая y = a1*x^2 + a2*x + a3 лучше приближает исходные данные")
    else:
        print("Так как погрешность f1(x) < f2(x), то прямая y = a*x + b лучше приближает исходные данные")


# исходные данные
x = [4.302, 4.381, 4.626, 4.886, 4.808, 4.872, 4.382, 4.181, 4.483, 4.418]
'''y = [8.067, 9.681, 11.494, 13.321, 12.931, 13.451, 9.562, 8.066, 10.250, 9.643]
'''
'''x = [1.577, 1.538, 1.333, 1.847, 1.797, 1.910, 1.371, 1.527, 1.632, 1.034]
y = [2.000, 2.397, 2.264, 1.987, 2.266, 1.837, 2.339, 2.260, 1.928, 2.819]'''

y = [5.861, 6.212, 2.868, 2.647, 6.198, 3.499, 3.529, 6.511, 5.955, 4.185]

A1, B1 = linear_dependence(x, y)
A2, B2 = quadratic_dependence(x, y)

print("Линейная функция")
c_1 = Gauss(A1, B1).GaussMethod()
d = {'a1': [c_1[0]], 'a2': [c_1[1]]}
df = pd.DataFrame(data=d)
print(df)

print("\nКвадратичная функция")
c_2 = Gauss(A2, B2).GaussMethod()
d = {'a1': [c_2[0]], 'a2': [c_2[1]], 'a3': [c_2[2]]}
df = pd.DataFrame(data=d)
print(df)

graph(x, y, c_1, c_2)

inaccuracy(x, y, c_1, c_2)
