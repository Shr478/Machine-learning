# Lab 6 - Locally weighted regression function

import numpy as np
import matplotlib.pyplot as plt
def gaussian_kernel(x, xi, tau):
    return np.exp(-np.sum((x - xi) ** 2) / (2 * tau ** 2))
def locally_weighted_regression(x_query, X, y, tau):
    m=X.shape[0]
    weights=np.array([gaussian_kernel(x_query, X[i], tau) for i in range(m)])
    W=np.diag(weights)
    X_transpose_W=X.T @ W
    theta=np.linalg.pinv(X_transpose_W @ X) @ X_transpose_W @ y
    
    return x_query @ theta
np.random.seed(42)
x=np.linspace(0, 2 * np.pi, 100)
y=np.sin(x) + 0.1 * np.random.randn(100)
X_bias=np.c_[np.ones(x.shape[0]), x]
x_test=np.linspace(0, 2 * np.pi, 200)
x_test_bias=np.c_[np.ones(x_test.shape[0]), x_test]

tau=0.5
y_pred=np.array([locally_weighted_regression(xi, X_bias, y, tau)for xi in x_test_bias])
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Training Data', alpha=0.7)
plt.plot(x_test, y_pred, color='blue', label=f'LWR Fit (tau={tau})', linewidth=2)

plt.xlabel('X',fontsize=12)
plt.ylabel('Y',fontsize=12)
plt.title('Locally Weighted Regression', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.show()
