# Модуль Neuron
import numpy as np


def onestep(x):
    b = 5
    if x >= b:
        return 1
    else:
        return 0


# Создание класса "Нейрон”
class Neuron:
    def __init__(self, w):
        self.w = w

    def y(self, x):
        s = np.dot(self.w, x)
        return onestep(s)


# Сумматор
# Суммируем входы
# функция активации
Xi = np.array([1, 0, 0, 1])  # Задание значении входам
Wi = np.array([5, 4, 3, 1])  # Веса входных сенсоров
n = Neuron(Wi)  # Создание объекта из класса Neuron
print("S= ", n.y(Xi))
