
# Linear Regression — Implementation Explanation

# Part I — English Version

## 1. Overview

The `LinearRegression` module implements the Ordinary Least Squares (OLS) algorithm using NumPy.

Two versions have been developed:

- **Version 1:** Pseudoinverse.
- **Version 2:** Batch Gradient Descent.

Both versions follow the same API:

```python
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X_new)
```

**Main attributes:**

| Attribute | Description |
|---|---|
| `fit_intercept` | Determines whether the model learns an intercept |
| `coef_` | Stores the learned feature weights |
| `intercept_` | Stores the learned intercept |

The `fit()` method trains the model, while the `predict()` method uses the learned parameters to generate predictions.

---

## 2. Version 1 — Pseudoinverse

### 2.1. Initialization

```python
def __init__(self, fit_intercept=True):
    self.fit_intercept = fit_intercept
    self.coef_ = None
    self.intercept_ = None
```

The constructor initializes the model configuration and the attributes used to store learned parameters.

Initially, `coef_` and `intercept_` are set to `None` because the model has not been trained.

After calling `fit()`, these attributes store the estimated weights and intercept.

### 2.2. Adding the Intercept Column

```python
if self.fit_intercept:
    X_b = np.c_[np.ones(len(X)), X]
else:
    X_b = X
```

`np.ones(len(X))` creates a vector of ones with a length equal to the number of samples.

`np.c_[]` concatenates this vector as the first column of the input matrix.

**Example:**

```python
X = [[1, 2],
     [2, 1]]
```

After adding the intercept column:

```python
X_b = [[1, 1, 2],
       [1, 2, 1]]
```

The first column represents the intercept term.

This transformation allows the intercept to be included in matrix multiplication.

If `fit_intercept=False`, the original input matrix is used without modification.

### 2.3. Solving Least Squares

```python
theta = np.linalg.pinv(X_b) @ y
```

Where:

- `np.linalg.pinv(X_b)`: Computes the Moore–Penrose pseudoinverse.
- `@`: Performs matrix multiplication.
- `theta`: Stores the estimated model parameters.

**Mathematical expression:**

$$
\boxed{\theta=X_b^+y}
$$

The pseudoinverse provides a least-squares solution even when the input matrix is rank-deficient.

This method does not require a learning rate or iterative training epochs.

### 2.4. Extracting Parameters

```python
if self.fit_intercept:
    self.intercept_ = theta[0]
    self.coef_ = theta[1:]
else:
    self.intercept_ = 0.0
    self.coef_ = theta
```

When `fit_intercept=True`, the parameter vector has the following structure:

```python
theta = [b, w1, w2, ...]
```

Therefore:

- `theta[0]`: Intercept.
- `theta[1:]`: Feature weights.

When `fit_intercept=False`, the entire parameter vector represents the feature weights.

The intercept is explicitly set to zero.

### 2.5. Returning the Model

```python
return self
```

The method returns the trained model instance.

This design supports method chaining:

```python
model = LinearRegression().fit(X, y)
```

**Important note:**

The training-state validation should be performed inside `predict()`, not at the end of `fit()`.

After successful training, the parameters have already been assigned.

---

## 3. Version 2 — Batch Gradient Descent

Version 2 replaces the Pseudoinverse solution with an iterative optimization process.

Instead of directly computing model parameters, Batch Gradient Descent repeatedly updates weights and intercept to minimize Mean Squared Error.

### 3.1. Input Conversion

```python
X = np.asarray(X, dtype=float)
y = np.asarray(y, dtype=float)
```

These statements convert the input data into NumPy arrays.

**Purposes:**

1. Enable matrix multiplication.
2. Support vectorized numerical operations.
3. Use floating-point values for calculations.

`np.asarray()` avoids unnecessary copying when the input is already a compatible NumPy array.

If conversion or type casting is required, a new array may be created.

### 3.2. Reading Dataset Dimensions

```python
m, n = X.shape
```

For a two-dimensional input matrix:

- `m`: Number of samples.
- `n`: Number of features.

**Example:**

```python
X.shape == (6, 2)
```

Therefore:

```python
m = 6
n = 2
```

The number of features determines the required length of the weight vector.

### 3.3. Parameter Initialization

```python
w = np.zeros(n)
b = 0.0
```

`np.zeros(n)` creates a weight vector containing `n` zeros.

**Example:**

```python
w = [0.0, 0.0]
```

The number of weights must equal the number of input features.

The intercept `b` is initialized to zero.

Zero initialization is valid for Linear Regression because its squared-error loss function is convex.

### 3.4. Training Loop

```python
for epoch in range(epochs):
```

The loop performs the optimization process for the specified number of epochs.

**Example:**

```python
epochs = 10000
```

In Batch Gradient Descent, this results in 10,000 parameter updates.

`range()` generates the sequence of loop indices because an integer cannot be iterated over directly.

### 3.5. Forward Pass

```python
y_pred = X @ w + b
```

This statement computes predictions for the entire training dataset.

**Mathematical expression:**

$$
\boxed{\hat{y}=Xw+b\mathbf{1}_m}
$$

**Array dimensions:**

```text
X:      (m, n)
w:      (n,)
y_pred: (m,)
```

NumPy automatically adds the scalar `b` to every element of the prediction vector through broadcasting.

### 3.6. Error Calculation

```python
error = y_pred - y
```

This statement computes the prediction error for every sample.

$$
\boxed{e=\hat{y}-y}
$$

The result is a vector containing `m` elements.

Each element represents the difference between a predicted value and its corresponding target value.

### 3.7. Weight Gradient

```python
dw = (2 / m) * (X.T @ error)
```

This statement calculates the gradient of Mean Squared Error with respect to the weights.

$$
\boxed{dw=\frac{2}{m}X^Te}
$$

`X.T` transposes the input matrix:

```text
(m, n) -> (n, m)
```

Matrix multiplication:

```text
(n, m) @ (m,) -> (n,)
```

The resulting `dw` vector contains `n` elements, corresponding to the `n` model weights.

Each element `dw[j]` represents the rate of change of the loss function with respect to the corresponding weight.

### 3.8. Intercept Gradient

```python
db = (2 / m) * np.sum(error)
```

This statement computes the gradient of Mean Squared Error with respect to the intercept.

$$
\boxed{db=\frac{2}{m}\sum_{i=1}^{m}e_i}
$$

`np.sum(error)` calculates the sum of all prediction errors.

The resulting `db` is a scalar value.

### 3.9. Parameter Update

```python
w -= learning_rate * dw
b -= learning_rate * db
```

These statements update the model parameters using Gradient Descent.

**Weight update:**

$$
\boxed{w\leftarrow w-\alpha dw}
$$

**Intercept update:**

$$
\boxed{b\leftarrow b-\alpha db}
$$

Where `learning_rate` corresponds to $\alpha$.

The learning rate controls the step size of each parameter update.

Both gradients are calculated before updating the parameters.

This ensures that the gradients are based on the same parameter values within each epoch.

### 3.10. Saving Parameters

```python
self.coef_ = w
self.intercept_ = b
return self
```

After completing the training loop:

- `coef_` stores the learned weight vector.
- `intercept_` stores the learned intercept.
- `return self` returns the trained model.

These parameters are subsequently used by `predict()`.

---

## 4. Prediction

```python
def predict(self, X):
    X = np.asarray(X, dtype=float)
    return X @ self.coef_ + self.intercept_
```

The `predict()` method performs two main operations:

1. Converts the input data into a NumPy array.
2. Calculates predictions using the learned model parameters.

**Prediction equation:**

$$
\boxed{\hat{y}=Xw+b\mathbf{1}}
$$

No gradient calculation or parameter update occurs during prediction.

**Example:**

```python
model.predict([[1, 3]])
```

Assume the model parameters are:

```python
coef_ = [2, 3]
intercept_ = 1
```

Then:

$$
\hat{y}=1(2)+3(3)+1=12
$$

**Output:**

```text
[12.]
```

---

## 5. Handling fit_intercept

In the initial Gradient Descent implementation, the intercept is always updated:

```python
b -= learning_rate * db
```

This causes `fit_intercept=False` to have no effect.

To fix this behavior, the intercept update must be conditional.

**Correct implementation:**

```python
w -= learning_rate * dw

if self.fit_intercept:
    b -= learning_rate * db
```

When `fit_intercept=False`, the intercept remains fixed at zero.

The model becomes:

$$
\boxed{\hat{y}=Xw}
$$

When `fit_intercept=True`, both weights and intercept are optimized.

---

## 6. Recommended Validation

The implementation should validate input data before training or prediction.

### 6.1. Input Shape

```python
if X.ndim != 2:
    raise ValueError("X must be a 2D array.")
```

This validation ensures that the input matrix has exactly two dimensions.

### 6.2. Sample Count

```python
if len(X) != len(y):
    raise ValueError(
        "X and y must contain the same number of samples."
    )
```

The number of input samples must match the number of target values.

### 6.3. Training State

Inside `predict()`:

```python
if self.coef_ is None:
    raise RuntimeError("Model has not been fitted yet.")
```

This prevents predictions from being generated before the model has been trained.

### 6.4. Feature Count

```python
if X.shape[1] != len(self.coef_):
    raise ValueError("Number of features does not match.")
```

The number of prediction features must match the number of features used during training.

### 6.5. Additional Validation

The implementation should also check:

- Empty datasets.
- Missing values (`NaN`).
- Infinite values (`inf`).
- Invalid learning rates.
- Invalid epoch counts.
- One-dimensional target arrays.
- Consistent feature dimensions.

These checks improve reliability and provide more informative error messages.

---

## 7. Testing

**Training dataset:**

```python
X = [
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 2],
    [5, 3],
    [6, 5]
]

y = [9, 8, 19, 15, 20, 28]
```

The dataset follows the linear relationship:

$$
\boxed{y=2x_1+3x_2+1}
$$

**Expected parameters after convergence:**

```text
coef_      ≈ [2.0, 3.0]
intercept_ ≈ 1.0
```

**Prediction example:**

```python
model.predict([[7, 2]])
```

Calculation:

$$
\hat{y}=2(7)+3(2)+1
$$

$$
\boxed{\hat{y}=21}
$$

**Expected output:**

```text
[21.]
```

### Comparing with Scikit-Learn

The custom implementation can be compared with Scikit-Learn.

```python
import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

reference = SklearnLinearRegression()
reference.fit(X, y)

print(reference.coef_)
print(reference.intercept_)
```

Numerical results can be compared using:

```python
np.allclose(custom_model.coef_, reference.coef_)
```

```python
np.allclose(
    custom_model.intercept_,
    reference.intercept_
)
```

`np.allclose()` allows small floating-point differences within specified tolerances.

For Gradient Descent, the comparison is meaningful only when the optimization has converged sufficiently.

---

## 8. Implementation Complexity

### 8.1. fit()

For Batch Gradient Descent, each epoch performs the following operations:

| Operation | Time Complexity |
|---|---|
| Matrix-vector multiplication | $O(mn)$ |
| Error calculation | $O(m)$ |
| Gradient calculation | $O(mn)$ |
| Parameter update | $O(n)$ |

With $k$ epochs, total training complexity is:

$$
\boxed{O(kmn)}
$$

Where:

- $m$: Number of samples.
- $n$: Number of features.
- $k$: Number of epochs.

### 8.2. predict()

For $m$ input samples and $n$ features, prediction requires matrix-vector multiplication.

$$
\boxed{O(mn)}
$$

No iterative optimization is performed during prediction.

---

## 9. Known Limitations

The current implementation has several limitations:

1. Supports only one-dimensional target vectors.
2. Does not perform complete input validation.
3. Does not fully validate training state before prediction.
4. Uses a fixed number of epochs.
5. Does not implement Early Stopping.
6. Does not store Loss History.
7. Does not automatically adjust the learning rate.
8. May diverge when the learning rate is too large.

Possible future improvements include:

- Input validation.
- Early Stopping.
- Loss History.
- Feature Scaling.
- Learning rate scheduling.
- Multi-output regression support.
- Convergence tolerance.

---

## 10. Summary

Both implementations use the same linear model:

$$
\boxed{\hat{y}=Xw+b\mathbf{1}_m}
$$

**Version 1 — Pseudoinverse**

Solves the least-squares problem directly using the Moore–Penrose Pseudoinverse.

$$
\boxed{\theta=X_b^+y}
$$

**Version 2 — Batch Gradient Descent**

Solves the least-squares problem by iteratively updating model parameters.

**Core implementation:**

```python
y_pred = X @ w + b
error = y_pred - y

dw = (2 / m) * (X.T @ error)
db = (2 / m) * np.sum(error)

w -= learning_rate * dw
b -= learning_rate * db
```

These statements correspond to four stages:

**Forward Pass → Error Calculation → Gradient Calculation → Parameter Update**

Both approaches aim to minimize the same Ordinary Least Squares objective, but they differ in their computational mechanisms.

---

# Part II — Phiên bản tiếng Việt

## 1. Tổng quan

Module `LinearRegression` triển khai thuật toán Ordinary Least Squares (OLS) bằng thư viện NumPy.

Hai phiên bản đã được xây dựng:

- **Version 1:** Pseudoinverse (Ma trận giả nghịch đảo).
- **Version 2:** Batch Gradient Descent (Gradient Descent toàn bộ tập dữ liệu).

Cả hai phiên bản sử dụng cùng một API:

```python
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X_new)
```

**Các thuộc tính chính:**

| Thuộc tính | Mô tả |
|---|---|
| `fit_intercept` | Quyết định mô hình có học hệ số tự do hay không |
| `coef_` | Lưu các trọng số đã học của những đặc trưng |
| `intercept_` | Lưu hệ số tự do đã học |

Phương thức `fit()` được sử dụng để huấn luyện mô hình, còn `predict()` sử dụng các tham số đã học để tạo giá trị dự đoán.

---

## 2. Phiên bản 1 — Pseudoinverse

### 2.1. Khởi tạo

```python
def __init__(self, fit_intercept=True):
    self.fit_intercept = fit_intercept
    self.coef_ = None
    self.intercept_ = None
```

Hàm khởi tạo thiết lập cấu hình mô hình và các thuộc tính dùng để lưu tham số.

Ban đầu, `coef_` và `intercept_` được đặt bằng `None` vì mô hình chưa được huấn luyện.

Sau khi gọi `fit()`, hai thuộc tính này sẽ lưu các trọng số và hệ số tự do đã ước lượng.

### 2.2. Thêm cột hệ số tự do

```python
if self.fit_intercept:
    X_b = np.c_[np.ones(len(X)), X]
else:
    X_b = X
```

`np.ones(len(X))` tạo một vector toàn số 1 có độ dài bằng số lượng mẫu dữ liệu.

`np.c_[]` ghép vector này thành cột đầu tiên của ma trận đầu vào.

**Ví dụ:**

```python
X = [[1, 2],
     [2, 1]]
```

Sau khi thêm cột hệ số tự do:

```python
X_b = [[1, 1, 2],
       [1, 2, 1]]
```

Cột đầu tiên biểu diễn hệ số tự do (Intercept).

Phép biến đổi này cho phép đưa intercept vào phép nhân ma trận.

Nếu `fit_intercept=False`, ma trận đầu vào ban đầu được sử dụng mà không thay đổi.

### 2.3. Giải bài toán bình phương tối thiểu

```python
theta = np.linalg.pinv(X_b) @ y
```

Trong đó:

- `np.linalg.pinv(X_b)`: Tính ma trận giả nghịch đảo Moore–Penrose.
- `@`: Thực hiện phép nhân ma trận.
- `theta`: Lưu vector tham số được ước lượng.

**Công thức toán học:**

$$
\boxed{\theta=X_b^+y}
$$

Pseudoinverse có thể tìm nghiệm bình phương tối thiểu ngay cả khi ma trận đầu vào không có hạng đầy đủ.

Phương pháp này không yêu cầu Learning Rate hoặc quá trình huấn luyện lặp theo epochs.

### 2.4. Trích xuất tham số

```python
if self.fit_intercept:
    self.intercept_ = theta[0]
    self.coef_ = theta[1:]
else:
    self.intercept_ = 0.0
    self.coef_ = theta
```

Khi `fit_intercept=True`, vector tham số có cấu trúc:

```python
theta = [b, w1, w2, ...]
```

Do đó:

- `theta[0]`: Hệ số tự do.
- `theta[1:]`: Các trọng số của đặc trưng.

Khi `fit_intercept=False`, toàn bộ vector tham số biểu diễn các trọng số.

Hệ số tự do được gán bằng 0.

### 2.5. Trả về mô hình

```python
return self
```

Phương thức trả về chính đối tượng mô hình sau khi huấn luyện.

Cách thiết kế này hỗ trợ viết liên tiếp các phương thức:

```python
model = LinearRegression().fit(X, y)
```

**Lưu ý quan trọng:**

Việc kiểm tra trạng thái huấn luyện nên được thực hiện trong `predict()`, không phải ở cuối `fit()`.

Sau khi quá trình huấn luyện thành công, các tham số đã được gán giá trị.

---

## 3. Phiên bản 2 — Batch Gradient Descent

Version 2 thay thế phép giải Pseudoinverse bằng một quá trình tối ưu hóa lặp.

Thay vì tính trực tiếp các tham số mô hình, Batch Gradient Descent liên tục cập nhật trọng số và hệ số tự do để tối thiểu hóa Mean Squared Error.

### 3.1. Chuyển đổi dữ liệu đầu vào

```python
X = np.asarray(X, dtype=float)
y = np.asarray(y, dtype=float)
```

Hai câu lệnh này chuyển dữ liệu đầu vào thành NumPy arrays.

**Mục đích:**

1. Hỗ trợ phép nhân ma trận.
2. Hỗ trợ các phép tính số học được vector hóa.
3. Sử dụng kiểu số thực để tính toán.

`np.asarray()` tránh sao chép dữ liệu không cần thiết khi đầu vào đã là NumPy array tương thích.

Nếu cần chuyển đổi kiểu dữ liệu hoặc định dạng, một array mới có thể được tạo ra.

### 3.2. Xác định kích thước dữ liệu

```python
m, n = X.shape
```

Với ma trận đầu vào hai chiều:

- `m`: Số lượng mẫu dữ liệu.
- `n`: Số lượng đặc trưng.

**Ví dụ:**

```python
X.shape == (6, 2)
```

Khi đó:

```python
m = 6
n = 2
```

Số lượng đặc trưng quyết định số phần tử cần có trong vector trọng số.

### 3.3. Khởi tạo tham số

```python
w = np.zeros(n)
b = 0.0
```

`np.zeros(n)` tạo một vector trọng số gồm `n` phần tử bằng 0.

**Ví dụ:**

```python
w = [0.0, 0.0]
```

Số lượng trọng số phải bằng số lượng đặc trưng đầu vào.

Hệ số tự do `b` được khởi tạo bằng 0.

Khởi tạo bằng 0 là hợp lệ đối với Linear Regression vì hàm mất mát bình phương sai số là hàm lồi.

### 3.4. Vòng lặp huấn luyện

```python
for epoch in range(epochs):
```

Vòng lặp thực hiện quá trình tối ưu hóa theo số epochs được chỉ định.

**Ví dụ:**

```python
epochs = 10000
```

Trong Batch Gradient Descent, điều này tương ứng với 10.000 lần cập nhật tham số.

`range()` tạo một dãy chỉ số cho vòng lặp vì không thể trực tiếp lặp qua một số nguyên.

### 3.5. Tính giá trị dự đoán (Forward Pass)

```python
y_pred = X @ w + b
```

Câu lệnh này tính các giá trị dự đoán trên toàn bộ tập dữ liệu huấn luyện.

**Công thức toán học:**

$$
\boxed{\hat{y}=Xw+b\mathbf{1}_m}
$$

**Kích thước các mảng:**

```text
X:      (m, n)
w:      (n,)
y_pred: (m,)
```

NumPy tự động cộng giá trị vô hướng `b` vào từng phần tử của vector dự đoán thông qua broadcasting.

### 3.6. Tính sai số

```python
error = y_pred - y
```

Câu lệnh này tính sai số dự đoán của từng mẫu dữ liệu.

$$
\boxed{e=\hat{y}-y}
$$

Kết quả là một vector chứa `m` phần tử.

Mỗi phần tử biểu diễn hiệu giữa giá trị dự đoán và giá trị thực tế tương ứng.

### 3.7. Tính Gradient của trọng số

```python
dw = (2 / m) * (X.T @ error)
```

Câu lệnh này tính gradient của Mean Squared Error theo các trọng số.

$$
\boxed{dw=\frac{2}{m}X^Te}
$$

`X.T` thực hiện chuyển vị ma trận đầu vào:

```text
(m, n) -> (n, m)
```

Phép nhân ma trận:

```text
(n, m) @ (m,) -> (n,)
```

Vector `dw` thu được chứa `n` phần tử, tương ứng với `n` trọng số của mô hình.

Mỗi phần tử `dw[j]` biểu diễn tốc độ thay đổi của hàm mất mát theo trọng số tương ứng.

### 3.8. Tính Gradient của hệ số tự do

```python
db = (2 / m) * np.sum(error)
```

Câu lệnh này tính gradient của Mean Squared Error theo hệ số tự do.

$$
\boxed{db=\frac{2}{m}\sum_{i=1}^{m}e_i}
$$

`np.sum(error)` tính tổng tất cả các sai số dự đoán.

Kết quả `db` là một giá trị vô hướng.

### 3.9. Cập nhật tham số

```python
w -= learning_rate * dw
b -= learning_rate * db
```

Hai câu lệnh này cập nhật các tham số mô hình bằng Gradient Descent.

**Cập nhật trọng số:**

$$
\boxed{w\leftarrow w-\alpha dw}
$$

**Cập nhật hệ số tự do:**

$$
\boxed{b\leftarrow b-\alpha db}
$$

Trong đó `learning_rate` tương ứng với $\alpha$.

Learning Rate kiểm soát kích thước bước cập nhật tham số.

Cả hai gradient đều được tính trước khi cập nhật tham số.

Điều này đảm bảo các gradient được tính dựa trên cùng một bộ tham số trong mỗi epoch.

### 3.10. Lưu tham số

```python
self.coef_ = w
self.intercept_ = b
return self
```

Sau khi hoàn thành vòng lặp huấn luyện:

- `coef_` lưu vector trọng số đã học.
- `intercept_` lưu hệ số tự do đã học.
- `return self` trả về đối tượng mô hình đã huấn luyện.

Các tham số này được sử dụng trong phương thức `predict()`.

---

## 4. Dự đoán (Prediction)

```python
def predict(self, X):
    X = np.asarray(X, dtype=float)
    return X @ self.coef_ + self.intercept_
```

Phương thức `predict()` thực hiện hai thao tác chính:

1. Chuyển dữ liệu đầu vào thành NumPy array.
2. Tính giá trị dự đoán bằng các tham số mô hình đã học.

**Phương trình dự đoán:**

$$
\boxed{\hat{y}=Xw+b\mathbf{1}}
$$

Không có bước tính gradient hoặc cập nhật tham số trong quá trình dự đoán.

**Ví dụ:**

```python
model.predict([[1, 3]])
```

Giả sử các tham số mô hình là:

```python
coef_ = [2, 3]
intercept_ = 1
```

Khi đó:

$$
\hat{y}=1(2)+3(3)+1=12
$$

**Kết quả:**

```text
[12.]
```

---

## 5. Xử lý fit_intercept

Trong phiên bản Gradient Descent ban đầu, hệ số tự do luôn được cập nhật:

```python
b -= learning_rate * db
```

Điều này khiến `fit_intercept=False` không có tác dụng.

Để khắc phục, việc cập nhật hệ số tự do cần được thực hiện có điều kiện.

**Cách triển khai đúng:**

```python
w -= learning_rate * dw

if self.fit_intercept:
    b -= learning_rate * db
```

Khi `fit_intercept=False`, hệ số tự do được giữ cố định bằng 0.

Mô hình khi đó có dạng:

$$
\boxed{\hat{y}=Xw}
$$

Khi `fit_intercept=True`, cả trọng số và hệ số tự do đều được tối ưu hóa.

---

## 6. Kiểm tra dữ liệu đầu vào

Implementation nên kiểm tra dữ liệu trước khi thực hiện huấn luyện hoặc dự đoán.

### 6.1. Kiểm tra số chiều đầu vào

```python
if X.ndim != 2:
    raise ValueError("X must be a 2D array.")
```

Điều kiện này đảm bảo ma trận đầu vào có đúng hai chiều.

### 6.2. Kiểm tra số lượng mẫu

```python
if len(X) != len(y):
    raise ValueError(
        "X and y must contain the same number of samples."
    )
```

Số lượng mẫu đầu vào phải bằng số lượng giá trị nhãn tương ứng.

### 6.3. Kiểm tra trạng thái huấn luyện

Trong `predict()`:

```python
if self.coef_ is None:
    raise RuntimeError("Model has not been fitted yet.")
```

Điều kiện này ngăn việc dự đoán khi mô hình chưa được huấn luyện.

### 6.4. Kiểm tra số lượng đặc trưng

```python
if X.shape[1] != len(self.coef_):
    raise ValueError("Number of features does not match.")
```

Số lượng đặc trưng khi dự đoán phải bằng số lượng đặc trưng được sử dụng trong quá trình huấn luyện.

### 6.5. Các kiểm tra bổ sung

Implementation cũng nên kiểm tra:

- Tập dữ liệu rỗng.
- Giá trị thiếu (`NaN`).
- Giá trị vô cực (`inf`).
- Learning Rate không hợp lệ.
- Số epochs không hợp lệ.
- Vector nhãn không phải một chiều.
- Sự nhất quán về số lượng đặc trưng.

Những kiểm tra này giúp chương trình hoạt động ổn định hơn và cung cấp thông báo lỗi dễ hiểu.

---

## 7. Kiểm thử (Testing)

**Tập dữ liệu huấn luyện:**

```python
X = [
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 2],
    [5, 3],
    [6, 5]
]

y = [9, 8, 19, 15, 20, 28]
```

Dữ liệu tuân theo quan hệ tuyến tính:

$$
\boxed{y=2x_1+3x_2+1}
$$

**Các tham số kỳ vọng sau khi hội tụ:**

```text
coef_      ≈ [2.0, 3.0]
intercept_ ≈ 1.0
```

**Ví dụ dự đoán:**

```python
model.predict([[7, 2]])
```

Tính toán:

$$
\hat{y}=2(7)+3(2)+1
$$

$$
\boxed{\hat{y}=21}
$$

**Kết quả kỳ vọng:**

```text
[21.]
```

### So sánh với Scikit-Learn

Có thể so sánh implementation tự xây dựng với Scikit-Learn.

```python
import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

reference = SklearnLinearRegression()
reference.fit(X, y)

print(reference.coef_)
print(reference.intercept_)
```

Có thể so sánh kết quả số bằng:

```python
np.allclose(custom_model.coef_, reference.coef_)
```

```python
np.allclose(
    custom_model.intercept_,
    reference.intercept_
)
```

`np.allclose()` cho phép chấp nhận những sai khác nhỏ của số thực trong phạm vi dung sai được chỉ định.

Đối với Gradient Descent, phép so sánh chỉ có ý nghĩa khi quá trình tối ưu đã hội tụ đủ gần nghiệm.

---

## 8. Độ phức tạp của Implementation

### 8.1. fit()

Đối với Batch Gradient Descent, mỗi epoch thực hiện các phép toán sau:

| Phép toán | Độ phức tạp thời gian |
|---|---|
| Nhân ma trận và vector | $O(mn)$ |
| Tính sai số | $O(m)$ |
| Tính gradient | $O(mn)$ |
| Cập nhật tham số | $O(n)$ |

Với $k$ epochs, tổng độ phức tạp thời gian là:

$$
\boxed{O(kmn)}
$$

Trong đó:

- $m$: Số lượng mẫu.
- $n$: Số lượng đặc trưng.
- $k$: Số epochs.

### 8.2. predict()

Với $m$ mẫu đầu vào và $n$ đặc trưng, quá trình dự đoán yêu cầu phép nhân ma trận và vector.

$$
\boxed{O(mn)}
$$

Không có quá trình tối ưu hóa lặp trong `predict()`.

---

## 9. Những hạn chế hiện tại

Implementation hiện tại có một số hạn chế:

1. Chỉ hỗ trợ vector nhãn một chiều.
2. Chưa kiểm tra đầy đủ dữ liệu đầu vào.
3. Chưa kiểm tra hoàn chỉnh trạng thái huấn luyện trước khi dự đoán.
4. Sử dụng số epochs cố định.
5. Chưa triển khai Early Stopping.
6. Chưa lưu Loss History.
7. Chưa có cơ chế tự động điều chỉnh Learning Rate.
8. Có thể phân kỳ khi Learning Rate quá lớn.

**Một số hướng cải thiện trong tương lai:**

- Kiểm tra dữ liệu đầu vào.
- Early Stopping.
- Loss History.
- Feature Scaling.
- Learning Rate Scheduling.
- Hỗ trợ hồi quy nhiều đầu ra.
- Điều kiện dừng dựa trên mức độ hội tụ.

---

## 10. Tổng kết

Cả hai implementation đều sử dụng cùng một mô hình tuyến tính:

$$
\boxed{\hat{y}=Xw+b\mathbf{1}_m}
$$

**Version 1 — Pseudoinverse**

Giải trực tiếp bài toán bình phương tối thiểu bằng ma trận giả nghịch đảo Moore–Penrose.

$$
\boxed{\theta=X_b^+y}
$$

**Version 2 — Batch Gradient Descent**

Giải bài toán bình phương tối thiểu bằng cách cập nhật lặp các tham số mô hình.

**Phần triển khai cốt lõi:**

```python
y_pred = X @ w + b
error = y_pred - y

dw = (2 / m) * (X.T @ error)
db = (2 / m) * np.sum(error)

w -= learning_rate * dw
b -= learning_rate * db
```

Các câu lệnh trên tương ứng với bốn giai đoạn:

**Forward Pass → Error Calculation → Gradient Calculation → Parameter Update**

Cả hai phương pháp đều hướng đến việc tối thiểu hóa cùng một hàm mục tiêu Ordinary Least Squares, nhưng khác nhau về cơ chế tính toán.
