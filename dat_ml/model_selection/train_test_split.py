import random
import math


def train_test_split(X, y, test_size=0.2, random_state=42):
    if len(X) != len(y):
        raise ValueError("X and y do not belong to the same sample set!")
    if len(X) == 0 or len(y) == 0:
        raise ValueError("X or y is empty!")
    if test_size <= 0 or test_size >= 1:
        raise ValueError("Invalid test_size value!")
    seed = random.Random(random_state)
    n = len(X)
    indices = list(range(n))
    seed.shuffle(indices)
    n_train = n - math.ceil(test_size * n)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test