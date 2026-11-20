
# Linear Regression — Theory

# Part I — English Version

## 1. Overview

Linear Regression is a supervised learning algorithm used to predict continuous numerical values based on input features.

The model assumes that predictions can be represented as a linear combination of input features.

**For a single feature:**

$$
\hat{y} = wx + b
$$

**For multiple features:**

$$
\hat{y} = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
$$

Where:

- $x_i$: The $i$-th input feature.
- $w_i$: Weight associated with the $i$-th feature.
- $b$: Intercept or bias term.
- $\hat{y}$: Predicted value.

**Objective:** Find the optimal weights and intercept that minimize prediction errors on the training dataset.

---

## 2. Mathematical Representation

Assume a dataset contains:

- $m$ samples.
- $n$ features.

**Input matrix:**

$$
X \in \mathbb{R}^{m \times n}
$$

**Weight vector:**

$$
w \in \mathbb{R}^{n}
$$

**Target vector:**

$$
y \in \mathbb{R}^{m}
$$

**Prediction equation:**

$$
\boxed{\hat{y} = Xw + b\mathbf{1}_m}
$$

Here, $\mathbf{1}_m$ is a vector of $m$ ones, ensuring that the intercept is added to every prediction.

In programming, broadcasting allows this operation to be written as `X @ w + b`.

### Example

```python
X = [
    [1, 2],
    [2, 1],
    [3, 4]
]

w = [2, 3]
b = 1
```

**Predictions:**

$$
\hat{y}_1 = 2(1) + 3(2) + 1 = 9
$$

$$
\hat{y}_2 = 2(2) + 3(1) + 1 = 8
$$

$$
\hat{y}_3 = 2(3) + 3(4) + 1 = 19
$$

**Result:**

$$
\boxed{\hat{y} = [9,8,19]^T}
$$

Each row of $X$ represents one sample, while each column represents one feature.

---

## 3. Loss Function

To measure the difference between predicted and actual values, Linear Regression commonly uses Mean Squared Error (MSE).

$$
\boxed{
MSE = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i-y_i)^2
}
$$

Where:

- $m$: Number of samples.
- $\hat{y}_i$: Predicted value.
- $y_i$: Actual value.

MSE calculates the average of squared prediction errors.

**Why squared errors?**

1. Eliminates negative error signs.
2. Penalizes large errors more heavily.
3. Produces a differentiable objective function.

**Optimization objective:**

$$
\boxed{\min_{w,b} MSE(w,b)}
$$

Minimizing this objective corresponds to the Ordinary Least Squares (OLS) problem.

---

## 4. Ordinary Least Squares

Ordinary Least Squares (OLS) estimates model parameters by minimizing the sum of squared residuals.

**Residual:**

$$
e_i = \hat{y}_i-y_i
$$

**OLS objective:**

$$
\boxed{
\min_{w,b}\sum_{i=1}^{m}e_i^2
}
$$

Because MSE differs from the sum of squared errors only by the positive constant $1/m$, both objectives have the same minimizers.

Two approaches are considered in this implementation:

1. **Pseudoinverse:** Solves the least-squares problem directly using matrix operations.
2. **Gradient Descent:** Solves the problem iteratively by updating parameters based on gradients.

Both methods aim to minimize the same objective function, but their computational mechanisms are different.

---

## 5. Method 1 — Pseudoinverse

### 5.1. Matrix Representation

To incorporate the intercept into matrix multiplication, add a column of ones to the input matrix.

$$
X_b =
\begin{bmatrix}
1 & x_{11} & \cdots & x_{1n}\\
1 & x_{21} & \cdots & x_{2n}\\
\vdots & \vdots & \ddots & \vdots\\
1 & x_{m1} & \cdots & x_{mn}
\end{bmatrix}
$$

Define the parameter vector:

$$
\theta =
\begin{bmatrix}
b\\
w_1\\
\vdots\\
w_n
\end{bmatrix}
$$

The prediction equation becomes:

$$
\boxed{\hat{y}=X_b\theta}
$$

Where:

$$
X_b\in\mathbb{R}^{m\times(n+1)}
$$

$$
\theta\in\mathbb{R}^{n+1}
$$

### 5.2. Normal Equation

The least-squares objective can be expressed as:

$$
J(\theta)=\frac{1}{m}\|X_b\theta-y\|_2^2
$$

Taking the gradient with respect to $\theta$:

$$
\nabla_\theta J
=
\frac{2}{m}X_b^T(X_b\theta-y)
$$

At an optimum, the gradient equals zero:

$$
X_b^T(X_b\theta-y)=0
$$

Rearranging:

$$
X_b^TX_b\theta=X_b^Ty
$$

If $X_b^TX_b$ is invertible:

$$
\boxed{
\theta=(X_b^TX_b)^{-1}X_b^Ty
}
$$

However, the matrix may be singular or nearly singular, making direct inversion impossible or numerically unstable.

### 5.3. Moore–Penrose Pseudoinverse

A more general solution uses the Moore–Penrose Pseudoinverse.

$$
\boxed{\theta=X_b^+y}
$$

Where $X_b^+$ denotes the pseudoinverse of $X_b$.

The pseudoinverse can be computed using Singular Value Decomposition (SVD).

$$
X_b=U\Sigma V^T
$$

$$
\boxed{X_b^+=V\Sigma^+U^T}
$$

The pseudoinverse provides a least-squares solution even when input features are linearly dependent.

Among multiple least-squares solutions, it selects the minimum Euclidean-norm solution.

**NumPy implementation:**

```python
theta = np.linalg.pinv(X_b) @ y
```

This method does not require a learning rate or multiple training epochs.

---

## 6. Method 2 — Gradient Descent

Gradient Descent is an iterative optimization algorithm that updates model parameters to reduce the loss function.

Unlike Pseudoinverse, Gradient Descent does not compute the solution directly.

Instead, it starts with initial parameter values and repeatedly moves in the direction opposite to the loss gradient.

### 6.1. Initialization

Initialize weights and intercept:

$$
w=[0,0,\ldots,0]^T
$$

$$
b=0
$$

For Linear Regression, zero initialization is valid because the loss function is convex.

### 6.2. Forward Pass

Calculate predictions using the current parameters.

$$
\boxed{\hat{y}=Xw+b\mathbf{1}_m}
$$

### 6.3. Error Calculation

Compute the prediction error vector.

$$
\boxed{e=\hat{y}-y}
$$

Each element represents the difference between a predicted value and its corresponding actual value.

### 6.4. Gradient Calculation

**Loss function:**

$$
J(w,b)=\frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i-y_i)^2
$$

**Gradient with respect to each weight:**

$$
\frac{\partial J}{\partial w_j}
=
\frac{2}{m}\sum_{i=1}^{m}x_{ij}e_i
$$

**Matrix form:**

$$
\boxed{dw=\frac{2}{m}X^Te}
$$

**Gradient with respect to the intercept:**

$$
\boxed{db=\frac{2}{m}\sum_{i=1}^{m}e_i}
$$

Where:

- $dw$: Gradient vector of weights.
- $db$: Gradient of the intercept.
- $X^T$: Transpose of $X$.
- $e$: Prediction error vector.

The gradient describes how the loss changes with respect to the model parameters.

### 6.5. Parameter Update

Update the weights:

$$
\boxed{w\leftarrow w-\alpha dw}
$$

Update the intercept:

$$
\boxed{b\leftarrow b-\alpha db}
$$

Where $\alpha$ is the learning rate.

The negative sign is used because the gradient points toward the steepest local increase in the loss function.

Moving in the opposite direction with a sufficiently small step reduces the loss.

### 6.6. Training Loop

**Training procedure:**

1. Initialize weights and intercept.
2. Calculate predictions.
3. Calculate prediction errors.
4. Calculate gradients.
5. Update model parameters.
6. Repeat until a stopping condition is reached.

In Batch Gradient Descent, one complete pass through the training dataset is called an epoch, and each epoch performs one parameter update.

---

## 7. Learning Rate and Epochs

### 7.1. Learning Rate

The learning rate controls the step size used to update model parameters.

- **Too small:** Slow convergence.
- **Appropriate:** Stable loss reduction.
- **Too large:** Oscillation or divergence.

There is no universally optimal learning rate for every dataset.

### 7.2. Epochs

An epoch represents one complete pass through the training dataset.

In Batch Gradient Descent, each epoch corresponds to one parameter update.

Increasing the number of epochs does not guarantee better results if the learning rate is inappropriate.

---

## 8. Fit Intercept

The `fit_intercept` parameter determines whether the model learns an intercept.

### 8.1. fit_intercept = True

The model learns both weights and intercept.

$$
\boxed{\hat{y}=Xw+b\mathbf{1}_m}
$$

For Pseudoinverse, a column of ones is added to the input matrix.

For Gradient Descent, the intercept is updated during training.

### 8.2. fit_intercept = False

The model learns only the weights.

$$
\boxed{\hat{y}=Xw}
$$

The intercept is fixed at zero.

$$
b=0
$$

For Pseudoinverse, the original input matrix is used without adding an intercept column.

For Gradient Descent, the intercept is not updated.

---

## 9. Prediction

After training, the model contains the learned parameters.

- `coef_`: Learned weight vector.
- `intercept_`: Learned intercept.

Prediction does not involve calculating gradients or retraining the model.

Instead, the learned parameters are applied to new input samples.

$$
\boxed{\hat{y}_{new}=X_{new}w+b\mathbf{1}}
$$

**NumPy example:**

```python
y_pred = X_new @ coef_ + intercept_
```

---

## 10. Comparison

| Criterion | Pseudoinverse | Gradient Descent |
|---|---|---|
| Approach | Direct least-squares solution | Iterative optimization |
| Learning Rate | Not required | Required |
| Epochs | Not required | Required |
| Convergence | Direct numerical solution | Depends on optimization |
| Numerical concerns | Conditioning and SVD tolerance | Learning rate and feature scaling |
| Main operation | Matrix decomposition | Gradient computation |
| Implementation | `np.linalg.pinv()` | Matrix multiplication and iteration |

Pseudoinverse is useful for studying direct least-squares solutions.

Gradient Descent is useful for understanding iterative optimization and parameter updates.

---

## 11. Computational Complexity

### 11.1. Gradient Descent

Assume:

- $m$: Number of samples.
- $n$: Number of features.
- $k$: Number of epochs.

For Batch Gradient Descent, the computational cost of one epoch is:

$$
O(mn)
$$

The total training complexity is:

$$
\boxed{O(kmn)}
$$

When using vectorized operations, additional memory is required for prediction vectors, error vectors, and gradients.

Excluding the storage of the input dataset, the additional memory complexity is generally:

$$
\boxed{O(m+n)}
$$

### 11.2. Pseudoinverse

The computational complexity depends on the matrix dimensions and decomposition algorithm.

Let:

$$
p=n+1
$$

For a dense matrix with $m\ge p$, an SVD-based pseudoinverse computation typically requires:

$$
\boxed{O(mp^2+p^3)}
$$

When $m$ is much larger than $p$, the dominant term is approximately:

$$
O(mp^2)
$$

For datasets with a large number of features, pseudoinverse computation can become expensive.

---

## 12. Limitations

Linear Regression has several limitations:

1. **Linearity:** The model is linear in its parameters.
2. **Sensitivity to outliers:** Squared errors give large residuals greater influence.
3. **Multicollinearity:** Strong correlations between features can make coefficient interpretation difficult.
4. **Learning rate dependency:** Gradient Descent requires an appropriate learning rate.
5. **Feature scale sensitivity:** Gradient Descent may converge slowly when features have significantly different scales.

Feature Scaling and Regularization can help address some of these limitations.

However, these techniques do not eliminate every limitation of Linear Regression.

---

## 13. Summary

Linear Regression estimates the parameters of a linear model by minimizing squared prediction errors.

The two implementation methods are:

### 13.1. Pseudoinverse

**Direct least-squares solution:**

$$
\boxed{\theta=X_b^+y}
$$

This method solves the least-squares problem using matrix operations without iterative gradient updates.

### 13.2. Gradient Descent

**Weight gradient:**

$$
\boxed{
dw=\frac{2}{m}X^T(Xw+b\mathbf{1}_m-y)
}
$$

**Intercept gradient:**

$$
\boxed{
db=\frac{2}{m}\sum_{i=1}^{m}(\hat{y}_i-y_i)
}
$$

**Weight update:**

$$
\boxed{w\leftarrow w-\alpha dw}
$$

**Intercept update:**

$$
\boxed{b\leftarrow b-\alpha db}
$$

This method approaches a least-squares solution through repeated parameter updates.

**Key takeaway:**

Although Pseudoinverse and Gradient Descent use different computational approaches, both aim to minimize the Ordinary Least Squares objective.

For Linear Regression, the squared-error objective is convex.

Therefore, with appropriate optimization settings and sufficient iterations, Batch Gradient Descent can approach a global least-squares solution.

---

# Part II — Phiên bản tiếng Việt

## 1. Tổng quan

Linear Regression (Hồi quy tuyến tính) là một thuật toán học có giám sát (Supervised Learning), được sử dụng để dự đoán các giá trị số liên tục dựa trên những đặc trưng đầu vào.

Mô hình giả định rằng giá trị dự đoán có thể được biểu diễn dưới dạng tổ hợp tuyến tính của các đặc trưng.

**Với một đặc trưng:**

$$
\hat{y} = wx + b
$$

**Với nhiều đặc trưng:**

$$
\hat{y} = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
$$

Trong đó:

- $x_i$: Đặc trưng đầu vào thứ $i$.
- $w_i$: Trọng số tương ứng với đặc trưng thứ $i$.
- $b$: Hệ số tự do (Intercept).
- $\hat{y}$: Giá trị dự đoán.

**Mục tiêu:** Tìm các trọng số và hệ số tự do tối ưu sao cho sai số dự đoán trên tập dữ liệu huấn luyện là nhỏ nhất.

---

## 2. Biểu diễn toán học

Giả sử một tập dữ liệu bao gồm:

- $m$ samples (mẫu dữ liệu).
- $n$ features (đặc trưng).

**Ma trận đầu vào:**

$$
X \in \mathbb{R}^{m \times n}
$$

**Vector trọng số:**

$$
w \in \mathbb{R}^{n}
$$

**Vector nhãn:**

$$
y \in \mathbb{R}^{m}
$$

**Phương trình dự đoán:**

$$
\boxed{\hat{y} = Xw + b\mathbf{1}_m}
$$

Trong đó, $\mathbf{1}_m$ là vector gồm $m$ phần tử bằng 1, giúp cộng hệ số tự do vào tất cả các giá trị dự đoán.

Trong lập trình, cơ chế broadcasting cho phép viết phép toán này đơn giản hơn dưới dạng `X @ w + b`.

### Ví dụ

```python
X = [
    [1, 2],
    [2, 1],
    [3, 4]
]

w = [2, 3]
b = 1
```

**Các giá trị dự đoán:**

$$
\hat{y}_1 = 2(1) + 3(2) + 1 = 9
$$

$$
\hat{y}_2 = 2(2) + 3(1) + 1 = 8
$$

$$
\hat{y}_3 = 2(3) + 3(4) + 1 = 19
$$

**Kết quả:**

$$
\boxed{\hat{y} = [9,8,19]^T}
$$

Mỗi hàng của ma trận $X$ đại diện cho một mẫu dữ liệu, còn mỗi cột đại diện cho một đặc trưng.

---

## 3. Hàm mất mát

Để đo lường mức độ sai lệch giữa giá trị dự đoán và giá trị thực tế, Linear Regression thường sử dụng hàm Mean Squared Error (MSE).

$$
\boxed{
MSE = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i-y_i)^2
}
$$

Trong đó:

- $m$: Số lượng mẫu.
- $\hat{y}_i$: Giá trị dự đoán.
- $y_i$: Giá trị thực tế.

MSE tính trung bình cộng của bình phương các sai số dự đoán.

**Tại sao sử dụng bình phương sai số?**

1. Loại bỏ dấu âm của sai số.
2. Phạt các sai số lớn mạnh hơn.
3. Tạo ra hàm mục tiêu khả vi, thuận tiện cho tối ưu hóa.

**Mục tiêu tối ưu:**

$$
\boxed{\min_{w,b} MSE(w,b)}
$$

Việc tối thiểu hóa hàm mục tiêu này tương ứng với bài toán bình phương tối thiểu thông thường (Ordinary Least Squares — OLS).

---

## 4. Phương pháp bình phương tối thiểu

Ordinary Least Squares (OLS) ước lượng các tham số mô hình bằng cách tối thiểu hóa tổng bình phương phần dư.

**Phần dư:**

$$
e_i = \hat{y}_i-y_i
$$

**Hàm mục tiêu OLS:**

$$
\boxed{
\min_{w,b}\sum_{i=1}^{m}e_i^2
}
$$

Do MSE chỉ khác tổng bình phương sai số ở hệ số dương $1/m$, hai hàm mục tiêu có cùng nghiệm tối ưu.

Hai phương pháp được nghiên cứu trong phần triển khai này:

1. **Pseudoinverse:** Giải trực tiếp bài toán bình phương tối thiểu bằng các phép toán ma trận.
2. **Gradient Descent:** Giải bài toán bằng cách cập nhật tham số lặp dựa trên gradient.

Cả hai phương pháp đều hướng đến việc tối thiểu hóa cùng một hàm mục tiêu, nhưng cơ chế tính toán khác nhau.

---

## 5. Phương pháp 1 — Pseudoinverse

### 5.1. Biểu diễn ma trận

Để đưa hệ số tự do vào phép nhân ma trận, thêm một cột gồm toàn số 1 vào ma trận đầu vào.

$$
X_b =
\begin{bmatrix}
1 & x_{11} & \cdots & x_{1n}\\
1 & x_{21} & \cdots & x_{2n}\\
\vdots & \vdots & \ddots & \vdots\\
1 & x_{m1} & \cdots & x_{mn}
\end{bmatrix}
$$

Định nghĩa vector tham số:

$$
\theta =
\begin{bmatrix}
b\\
w_1\\
\vdots\\
w_n
\end{bmatrix}
$$

Phương trình dự đoán trở thành:

$$
\boxed{\hat{y}=X_b\theta}
$$

Trong đó:

$$
X_b\in\mathbb{R}^{m\times(n+1)}
$$

$$
\theta\in\mathbb{R}^{n+1}
$$

### 5.2. Phương trình chuẩn (Normal Equation)

Hàm mục tiêu bình phương tối thiểu có thể được biểu diễn dưới dạng:

$$
J(\theta)=\frac{1}{m}\|X_b\theta-y\|_2^2
$$

Lấy gradient của hàm mất mát theo $\theta$:

$$
\nabla_\theta J
=
\frac{2}{m}X_b^T(X_b\theta-y)
$$

Tại điểm tối ưu, gradient bằng 0:

$$
X_b^T(X_b\theta-y)=0
$$

Biến đổi lại:

$$
X_b^TX_b\theta=X_b^Ty
$$

Nếu $X_b^TX_b$ khả nghịch:

$$
\boxed{
\theta=(X_b^TX_b)^{-1}X_b^Ty
}
$$

Tuy nhiên, ma trận có thể suy biến hoặc gần suy biến, khiến việc tính nghịch đảo không thể thực hiện hoặc không ổn định về mặt số học.

### 5.3. Ma trận giả nghịch đảo Moore–Penrose

Một cách giải tổng quát hơn là sử dụng ma trận giả nghịch đảo Moore–Penrose (Moore–Penrose Pseudoinverse).

$$
\boxed{\theta=X_b^+y}
$$

Trong đó $X_b^+$ là ma trận giả nghịch đảo của $X_b$.

Ma trận giả nghịch đảo có thể được tính bằng phương pháp phân rã giá trị kỳ dị (Singular Value Decomposition — SVD).

$$
X_b=U\Sigma V^T
$$

$$
\boxed{X_b^+=V\Sigma^+U^T}
$$

Pseudoinverse có thể tìm nghiệm bình phương tối thiểu ngay cả khi các đặc trưng phụ thuộc tuyến tính.

Nếu tồn tại nhiều nghiệm bình phương tối thiểu, phương pháp này chọn nghiệm có chuẩn Euclid nhỏ nhất.

**Triển khai bằng NumPy:**

```python
theta = np.linalg.pinv(X_b) @ y
```

Phương pháp này không yêu cầu Learning Rate và không cần thực hiện nhiều epochs huấn luyện.

---

## 6. Phương pháp 2 — Gradient Descent

Gradient Descent là thuật toán tối ưu hóa lặp, cập nhật các tham số mô hình nhằm giảm dần giá trị hàm mất mát.

Khác với Pseudoinverse, Gradient Descent không tính nghiệm trực tiếp.

Thay vào đó, thuật toán bắt đầu từ các tham số khởi tạo và liên tục di chuyển theo hướng ngược với gradient của hàm mất mát.

### 6.1. Khởi tạo tham số

Khởi tạo trọng số và hệ số tự do:

$$
w=[0,0,\ldots,0]^T
$$

$$
b=0
$$

Đối với Linear Regression, khởi tạo bằng 0 là hợp lệ vì hàm mất mát có tính lồi.

### 6.2. Tính giá trị dự đoán (Forward Pass)

Tính các giá trị dự đoán bằng bộ tham số hiện tại.

$$
\boxed{\hat{y}=Xw+b\mathbf{1}_m}
$$

### 6.3. Tính sai số

Tính vector sai số dự đoán.

$$
\boxed{e=\hat{y}-y}
$$

Mỗi phần tử biểu diễn sự chênh lệch giữa giá trị dự đoán và giá trị thực tế tương ứng.

### 6.4. Tính Gradient

**Hàm mất mát:**

$$
J(w,b)=\frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i-y_i)^2
$$

**Gradient theo từng trọng số:**

$$
\frac{\partial J}{\partial w_j}
=
\frac{2}{m}\sum_{i=1}^{m}x_{ij}e_i
$$

**Dạng ma trận:**

$$
\boxed{dw=\frac{2}{m}X^Te}
$$

**Gradient theo hệ số tự do:**

$$
\boxed{db=\frac{2}{m}\sum_{i=1}^{m}e_i}
$$

Trong đó:

- $dw$: Vector gradient của các trọng số.
- $db$: Gradient của hệ số tự do.
- $X^T$: Ma trận chuyển vị của $X$.
- $e$: Vector sai số dự đoán.

Gradient biểu diễn mức độ thay đổi của hàm mất mát khi các tham số mô hình thay đổi.

### 6.5. Cập nhật tham số

Cập nhật các trọng số:

$$
\boxed{w\leftarrow w-\alpha dw}
$$

Cập nhật hệ số tự do:

$$
\boxed{b\leftarrow b-\alpha db}
$$

Trong đó $\alpha$ là tốc độ học (Learning Rate).

Dấu trừ được sử dụng vì gradient chỉ hướng tăng nhanh nhất của hàm mất mát tại điểm hiện tại.

Di chuyển theo hướng ngược lại với bước đủ nhỏ sẽ giúp giảm giá trị hàm mất mát.

### 6.6. Vòng lặp huấn luyện

**Quy trình huấn luyện:**

1. Khởi tạo trọng số và hệ số tự do.
2. Tính giá trị dự đoán.
3. Tính sai số dự đoán.
4. Tính gradient.
5. Cập nhật các tham số mô hình.
6. Lặp lại cho đến khi đạt điều kiện dừng.

Trong Batch Gradient Descent, một lần sử dụng toàn bộ tập dữ liệu huấn luyện được gọi là một epoch, và mỗi epoch thực hiện một lần cập nhật tham số.

---

## 7. Learning Rate và Epochs

### 7.1. Learning Rate

Learning Rate kiểm soát kích thước bước cập nhật các tham số mô hình.

- **Quá nhỏ:** Hội tụ chậm.
- **Phù hợp:** Hàm mất mát giảm ổn định.
- **Quá lớn:** Có thể gây dao động hoặc phân kỳ.

Không tồn tại một Learning Rate tối ưu chung cho mọi tập dữ liệu.

### 7.2. Epochs

Một epoch đại diện cho một lần thuật toán duyệt qua toàn bộ tập dữ liệu huấn luyện.

Trong Batch Gradient Descent, mỗi epoch tương ứng với một lần cập nhật tham số.

Tăng số lượng epochs không đảm bảo kết quả tốt hơn nếu Learning Rate không phù hợp.

---

## 8. Fit Intercept

Tham số `fit_intercept` quyết định mô hình có học hệ số tự do hay không.

### 8.1. fit_intercept = True

Mô hình học cả trọng số và hệ số tự do.

$$
\boxed{\hat{y}=Xw+b\mathbf{1}_m}
$$

Với Pseudoinverse, một cột toàn số 1 được thêm vào ma trận đầu vào.

Với Gradient Descent, hệ số tự do được cập nhật trong quá trình huấn luyện.

### 8.2. fit_intercept = False

Mô hình chỉ học các trọng số.

$$
\boxed{\hat{y}=Xw}
$$

Hệ số tự do được cố định bằng 0.

$$
b=0
$$

Với Pseudoinverse, sử dụng ma trận đầu vào ban đầu mà không thêm cột hệ số tự do.

Với Gradient Descent, hệ số tự do không được cập nhật.

---

## 9. Dự đoán (Prediction)

Sau khi huấn luyện, mô hình lưu trữ các tham số đã học.

- `coef_`: Vector trọng số đã học.
- `intercept_`: Hệ số tự do đã học.

Quá trình dự đoán không yêu cầu tính lại gradient hoặc huấn luyện lại mô hình.

Thay vào đó, các tham số đã học được áp dụng lên các mẫu dữ liệu đầu vào mới.

$$
\boxed{\hat{y}_{new}=X_{new}w+b\mathbf{1}}
$$

**Ví dụ NumPy:**

```python
y_pred = X_new @ coef_ + intercept_
```

---

## 10. So sánh hai phương pháp

| Tiêu chí | Pseudoinverse | Gradient Descent |
|---|---|---|
| Cách tiếp cận | Giải trực tiếp bài toán bình phương tối thiểu | Tối ưu hóa lặp |
| Learning Rate | Không cần | Cần |
| Epochs | Không cần | Cần |
| Hội tụ | Tính nghiệm số trực tiếp | Phụ thuộc quá trình tối ưu |
| Vấn đề số học | Điều kiện ma trận và ngưỡng SVD | Tốc độ học và chuẩn hóa đặc trưng |
| Phép toán chính | Phân rã ma trận | Tính gradient |
| Triển khai | `np.linalg.pinv()` | Nhân ma trận và lặp |

Pseudoinverse phù hợp để nghiên cứu cách tìm nghiệm bình phương tối thiểu trực tiếp.

Gradient Descent phù hợp để nghiên cứu quá trình tối ưu hóa lặp và cơ chế cập nhật tham số.

---

## 11. Độ phức tạp tính toán

### 11.1. Gradient Descent

Giả sử:

- $m$: Số mẫu.
- $n$: Số đặc trưng.
- $k$: Số epochs.

Đối với Batch Gradient Descent, chi phí tính toán của một epoch là:

$$
O(mn)
$$

Độ phức tạp thời gian tổng cộng của quá trình huấn luyện là:

$$
\boxed{O(kmn)}
$$

Khi sử dụng các phép toán vector hóa, thuật toán cần thêm bộ nhớ để lưu vector dự đoán, vector sai số và các gradient.

Không tính bộ nhớ lưu trữ tập dữ liệu đầu vào, độ phức tạp bộ nhớ bổ sung thường là:

$$
\boxed{O(m+n)}
$$

### 11.2. Pseudoinverse

Độ phức tạp tính toán phụ thuộc vào kích thước ma trận và thuật toán phân rã được sử dụng.

Đặt:

$$
p=n+1
$$

Với ma trận dense có $m\ge p$, việc tính pseudoinverse bằng SVD thường có độ phức tạp thời gian:

$$
\boxed{O(mp^2+p^3)}
$$

Khi $m$ lớn hơn nhiều so với $p$, thành phần chi phối xấp xỉ:

$$
O(mp^2)
$$

Với các tập dữ liệu có số lượng đặc trưng lớn, việc tính pseudoinverse có thể tiêu tốn nhiều tài nguyên.

---

## 12. Hạn chế

Linear Regression có một số hạn chế nhất định:

1. **Tính tuyến tính:** Mô hình tuyến tính theo các tham số.
2. **Nhạy với giá trị ngoại lai:** Việc bình phương sai số khiến những phần dư lớn có ảnh hưởng mạnh hơn.
3. **Đa cộng tuyến:** Các đặc trưng tương quan mạnh có thể gây khó khăn khi diễn giải các hệ số.
4. **Phụ thuộc Learning Rate:** Gradient Descent cần một tốc độ học phù hợp.
5. **Nhạy với thang đo đặc trưng:** Gradient Descent có thể hội tụ chậm khi các đặc trưng có thang đo chênh lệch lớn.

Feature Scaling và Regularization có thể giúp khắc phục một số hạn chế trên.

Tuy nhiên, các kỹ thuật này không loại bỏ được toàn bộ hạn chế của Linear Regression.

---

## 13. Tổng kết

Linear Regression ước lượng các tham số của mô hình tuyến tính bằng cách tối thiểu hóa bình phương sai số dự đoán.

Hai phương pháp triển khai bao gồm:

### 13.1. Pseudoinverse

**Nghiệm bình phương tối thiểu trực tiếp:**

$$
\boxed{\theta=X_b^+y}
$$

Phương pháp này giải bài toán bình phương tối thiểu thông qua các phép toán ma trận mà không cần cập nhật gradient lặp.

### 13.2. Gradient Descent

**Gradient của trọng số:**

$$
\boxed{
dw=\frac{2}{m}X^T(Xw+b\mathbf{1}_m-y)
}
$$

**Gradient của hệ số tự do:**

$$
\boxed{
db=\frac{2}{m}\sum_{i=1}^{m}(\hat{y}_i-y_i)
}
$$

**Cập nhật trọng số:**

$$
\boxed{w\leftarrow w-\alpha dw}
$$

**Cập nhật hệ số tự do:**

$$
\boxed{b\leftarrow b-\alpha db}
$$

Phương pháp này tiến dần đến nghiệm bình phương tối thiểu thông qua các lần cập nhật tham số liên tiếp.

**Kết luận chính:**

Mặc dù Pseudoinverse và Gradient Descent sử dụng các cơ chế tính toán khác nhau, cả hai đều hướng đến việc tối thiểu hóa hàm mục tiêu Ordinary Least Squares.

Đối với Linear Regression, hàm mục tiêu bình phương sai số là hàm lồi.

Vì vậy, với cấu hình tối ưu phù hợp và đủ số lần lặp, Batch Gradient Descent có thể tiến đến nghiệm tối ưu toàn cục của bài toán bình phương tối thiểu.
