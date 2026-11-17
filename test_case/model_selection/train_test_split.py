from dat_ml.model_selection import train_test_split

X = [
    [25, 1, 5],
    [30, 1, 8],
    [35, 2, 3],
    [40, 2, 10],
    [45, 2, 6],
    [50, 3, 4],
    [55, 3, 12],
    [60, 2, 7],
    [65, 3, 9],
    [70, 4, 2],
    [75, 3, 5],
    [80, 4, 11],
    [85, 4, 6],
    [90, 3, 8],
    [95, 5, 4],
    [100, 4, 10],
    [105, 5, 7],
    [110, 5, 3],
    [115, 4, 9],
    [120, 5, 6]
]

y = [
    550, 620, 750, 800, 900,
    1050, 1100, 1200, 1350, 1500,
    1550, 1700, 1800, 1900, 2050,
    2100, 2250, 2350, 2400, 2550
]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=1.1,
    random_state=42
)

print(f"X_train: {X_train}")
print(f"X_test: {X_test}")
print(f"y_train: {y_train}")
print(f"y_test: {y_test}")