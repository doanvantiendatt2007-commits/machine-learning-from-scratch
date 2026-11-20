from dat_ml.linear_model import LinearRegression

X = [
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 2],
    [5, 3],
    [6, 5]
]

y = [9, 8, 19, 14, 19, 26]

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict([[1, 3]])

print(
    f"y_pred = {y_pred}\n",
    f"coef_ = {model.coef_}\n"
    f"intercept_ = {model.intercept_}"
)