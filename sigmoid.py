# Модуль sigmoid
import numpy as np


# функция активации: f(x) = 1 / (1 + e^-x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Создание класса "Нейрон”
class Neuron:
    def __init__(self, w):
        self.w = w

    def y(self, x):
        s = np.dot(self.w, x)
        return sigmoid(s)


# Сумматор
# Суммируем входы
# функция активации
Xi = np.array([0, 0, 1, 1])  # Задание значении входам
Wi = np.array([5, 4, 3, 1])  # Веса входных сенсоров
n = Neuron(Wi)  # Создание объекта из класса Neuron
print("Y= ", n.y(Xi))
