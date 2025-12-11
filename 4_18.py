# Listing 4.18_1 с улучшенной загрузкой данных
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import requests


# Функция для загрузки данных с проверкой локального файла
def load_iris_data(
    url="https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv",
    local_filename="iris.data",
):
    """
    Загружает данные Iris с проверкой локального файла

    Параметры:
    url: str - URL для загрузки данных
    local_filename: str - имя локального файла для сохранения/чтения

    Возвращает:
    DataFrame с данными Iris

    Исключения:
    ConnectionError: если не удалось загрузить данные из интернета
    FileNotFoundError: если файл не найден локально и не удалось загрузить из интернета
    """

    # Создаем директорию для данных, если её нет
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    local_path = os.path.join(data_dir, local_filename)

    # Пытаемся загрузить из локального файла
    if os.path.exists(local_path):
        print(f"Загружаем данные из локального файла: {local_path}")
        try:
            df = pd.read_csv(local_path, header=None)
            print("✓ Данные успешно загружены из локального файла")
            return df
        except Exception as e:
            print(f"✗ Ошибка при чтении локального файла: {e}")
            print("Попытка загрузить из интернета...")
    else:
        print(f"Локальный файл {local_path} не найден")

    # Загружаем из интернета
    print(f"Загружаем данные из интернета: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Проверяем статус ответа

        # Сохраняем данные в файл
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(response.text)

        print(f"✓ Данные успешно загружены из интернета и сохранены в {local_path}")

        # Загружаем из сохраненного файла
        df = pd.read_csv(local_path, header=None)
        return df

    except requests.exceptions.ConnectionError:
        error_msg = f"Не удалось подключиться к {url}. Проверьте интернет-соединение."
        print(f"✗ {error_msg}")
        raise ConnectionError(error_msg)
    except requests.exceptions.Timeout:
        error_msg = f"Превышено время ожидания при загрузке из {url}"
        print(f"✗ {error_msg}")
        raise ConnectionError(error_msg)
    except requests.exceptions.RequestException as e:
        error_msg = f"Ошибка при загрузке данных из интернета: {e}"
        print(f"✗ {error_msg}")
        raise ConnectionError(error_msg)
    except Exception as e:
        error_msg = f"Неизвестная ошибка при загрузке данных: {e}"
        print(f"✗ {error_msg}")
        raise RuntimeError(error_msg)


# Описание класса Perceptron
class Perceptron(object):
    """
    Классификатор на основе персептрона.
    Параметры
    eta:float  - Темп обучения (между 0.0 и 1.0)
    n_iter:int - Проходы по тренировочному набору данных.
    Атрибуты
    w_: 1-мерный массив - Весовые коэффициенты после подгонки.
    errors_: список - Число случаев ошибочной классификации в каждой эпохе.
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    """
    Выполнить подгонку модели под тренировочные данные.
    Параметры
    X : массивоподобный, форма = [n_sam ples, n_features] тренировочные векторы, где 
                                    n_samples - число образцов и
                                    n _features - число признаков, 
    у : массивоподобный, форма = [n_samples] Целевые значения.
    Возвращает
    self: object
    """

    def fit(self, x, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    """Рассчитать чистый вход"""

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    """Вернуть метку класса после единичного скачка"""

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# Основная часть программы
try:
    print("=" * 60)
    print("Загрузка данных Iris для персептрона")
    print("=" * 60)

    # Загрузка данных с использованием новой функции
    df = load_iris_data()

    print("\nМассив данных Iris:")
    print(df.to_string())

    # выборка из объекта DF 100 элементов (столбец 4 название цветков) и загрузка его в одномерный массив Y и печать
    y = df.iloc[0:100, 4].values
    print("\nЗначение четвертого столбца Y - 100")
    print(y)

    # Преобразование названий цветков (столбец 4) в одномерный массив (вектор) из -1 и 1
    y = np.where(y == "Iris-setosa", -1, 1)
    print("\nЗначение названий цветков в виде -1 и 1, Y - 100")
    print(y)

    # выборка из объекта DF массива 100 элементов (столбец 0 и столбец 2), загрузка его в массив X (иатрица) и печать
    X = df.iloc[0:100, [0, 2]].values
    print("\nЗначение X - 100")
    print(X)
    print("Конец X")

    # Формирование параметров значений для вывода на график
    # Первые 50 элементов (Строки 0-50, столбцы 0,1)
    plt.scatter(X[0:50, 0], X[0:50, 1], color="red", marker="o", label="щетинистый")
    # Следующие 50 элементов (Строки 50-100, столбцы 0,1)
    plt.scatter(
        X[50:100, 0], X[50:100, 1], color="blue", marker="x", label="разноцветный"
    )

    # Формировние названий осей и вывод графика на экран
    plt.xlabel("длина чашелистика")
    plt.ylabel("длина лепестка")
    plt.legend(loc="upper left")
    plt.title("Визуализация данных Iris (первые 100 образцов)")
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\n✓ Программа успешно выполнена!")

except ConnectionError as e:
    print(f"\n✗ Критическая ошибка: {e}")
    print("Программа не может продолжить без данных.")
    print("Проверьте интернет-соединение и повторите попытку.")
except FileNotFoundError as e:
    print(f"\n✗ Ошибка: Файл не найден: {e}")
except Exception as e:
    print(f"\n✗ Неожиданная ошибка: {e}")
    import traceback

    traceback.print_exc()
finally:
    print("\n" + "=" * 60)
    print("Завершение работы программы")
    print("=" * 60)
