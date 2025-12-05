# Listing 4.14
# Модуль Urok22
import random
import matplotlib.pyplot as plt
import numpy as np

k = random.uniform(-5, 5)  # Коэффициент при x
c = random.uniform(-5, 5)  # Свободный член уравнения прямой
print("Начальная прямая: Y =  ", k, "* X + ", c)  # Вывод данных начальной прямой
rate = 0.0001  # Скорость обучения

# Набор точек X:Y
data = {
    22: 150,
    23: 155,
    24: 160,
    25: 162,
    26: 171,
    27: 174,
    28: 180,
    29: 183,
    30: 189,
    31: 192,
}


# Высчитать y
def proceed(x):
    return x * k + c


# Обучение сети
for i in range(99999):
    x = random.choice(list(data.keys()))
    true_result = data[x]
    out = proceed(x)
    delta = true_result - out
    k += delta * rate * x
    c += delta * rate

print(f"Готовая прямая: Y = {k:.4f} * X + {c:.4f}")

# Визуализация
plt.figure(figsize=(10, 6))

# Точки данных
x_vals = list(data.keys())
y_vals = list(data.values())
plt.scatter(x_vals, y_vals, color="blue", s=100, label="Исходные данные", zorder=5)

# Прямая после обучения
x_line = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 100)
y_line = k * x_line + c
plt.plot(x_line, y_line, "r-", linewidth=3, label=f"y = {k:.2f}x + {c:.2f}")

# Настройки графика
plt.xlabel("X", fontsize=12)
plt.ylabel("Y", fontsize=12)
plt.title("Линейная регрессия: аппроксимация данных прямой", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()

plt.show()
