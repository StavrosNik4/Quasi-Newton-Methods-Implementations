import numpy as np

from scipy.optimize import line_search
from scipy.linalg import cho_factor, cho_solve

import matplotlib.pyplot as plt

def objective_function(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (x[0] - 1) ** 2


def gradient(x):
    grad = np.zeros_like(x)
    grad[0] = -400 * x[0] * (x[1] - x[0] ** 2) + 2 * (x[0] - 1)
    grad[1] = 200 * (x[1] - x[0] ** 2)
    return grad


def sr1(f, grad, x0, max_iter=100, tol=1e-6, epsilon=1e-8):
    x_k = x0
    n = len(x0)
    B_k = np.eye(n)  # Initial Hessian approximation
    steps = 0
    x_values = [x_k.copy()]
    f_values = [f(x_k)]
    print(f"Step {steps}: x = {x_k}, f(x) = {f(x_k)}")
    for i in range(max_iter):
        grad_k = grad(x_k)
        # Stopping criterion
        if np.linalg.norm(grad_k) < tol:
            break

        # Perform Cholesky decomposition of B_k
        try:
            L, lower = cho_factor(B_k)
            p_k = cho_solve((L, lower), -grad_k)
        except np.linalg.LinAlgError:
            print("Cholesky decomposition failed. Using gradient descent direction.")
            p_k = -grad_k

        # Line search method to find step size α
        alpha_k = line_search(f, grad, x_k, p_k)[0]
        if alpha_k is None:
            alpha_k = 1e-4  # Default to a smaller step size if line search fails

        # Update x_k
        x_k1 = x_k + alpha_k * p_k

        # Compute s_k and y_k
        s_k = x_k1 - x_k
        y_k = grad(x_k1) - grad_k

        # Reshape s_k and y_k to column vectors
        s_k = s_k.reshape(-1, 1)
        y_k = y_k.reshape(-1, 1)

        # Calculate the SR1 update
        Bs = B_k @ s_k
        ys_diff = y_k - Bs
        denom = ys_diff.T @ s_k

        # Add condition to prevent division by zero or small values
        if np.abs(denom) > epsilon:
            B_k = B_k + (ys_diff @ ys_diff.T) / denom

        # Update x_k for the next iteration
        x_k = x_k1

        steps += 1
        x_values.append(x_k.copy())
        f_values.append(f(x_k))
        print(f"Step {steps}: x = {x_k}, f(x) = {f(x_k)}")
    return x_k, steps, x_values, f_values
x_values_scipy = []
f_values_scipy = []
scipy_steps = 0

def callback(xk, state):
    global scipy_steps
    scipy_steps += 1
    x_values_scipy.append(xk.copy())
    f_values_scipy.append(objective_function(xk))
    print(f"Step {scipy_steps}: x = {xk}, f(x) = {objective_function(xk)}")

# Initial guess
x0 = np.array([4.0, 4.0])

# Custom SR1
optimal_solution, steps, x_values_custom, f_values_custom = sr1(objective_function, gradient, x0)
print("Optimal solution:", optimal_solution)
print("Objective function value at optimal solution:", objective_function(optimal_solution))
print("Number of steps taken:", steps)
print()

# # SciPy SR1
# result = minimize(objective_function, x0, method='trust-constr', jac=gradient, callback=callback, options={'maxiter': 100, 'gtol': 1e-6})
# print("Optimal solution (scipy):", result.x)
# print("Objective function value at optimal solution (scipy):", result.fun)
# print("Number of steps taken (scipy):", scipy_steps)

# # Create a grid of points
# x = np.linspace(-10, 10, 4000)
# y = np.linspace(-10, 10, 4000)
# X, Y = np.meshgrid(x, y)
# Z = objective_function([X, Y])

# # Plotting the results
# plt.figure(figsize=(12, 6))
#
# # Contour plot of the objective function
# plt.contour(X, Y, Z, levels=10, cmap='viridis')
#
# # Custom SR1 Path
# x_values_custom = np.array(x_values_custom)
# plt.plot(x_values_custom[:, 0], x_values_custom[:, 1], 'r-o', label='Custom SR1')
#
# # # SciPy SR1 Path
# # x_values_scipy = np.array(x_values_scipy)
# # plt.plot(x_values_scipy[:, 0], x_values_scipy[:, 1], 'b-o', label='SciPy SR1')
#
# plt.title('Optimization Paths on Objective Function Contour')
# plt.xlabel('x[0]')
# plt.ylabel('x[1]')
# plt.legend()
# plt.show()

# Plotting

x_values_custom = np.array(x_values_custom)
f_values_custom = np.array(f_values_custom)

# Objective function surface
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x, y)
Z = objective_function([X, Y])

fig = plt.figure(figsize=(14, 6))

# 3D plot
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.plot(x_values_custom[:, 0], x_values_custom[:, 1], f_values_custom, 'r-o')
ax.scatter(x0[0], x0[1], objective_function(x0), color='blue', s=100, label='Initial Point')
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('f(x)')
ax.set_title('Objective Function Surface and Optimization Path')
ax.legend()

# Contour plot
ax2 = fig.add_subplot(122)
ax2.contour(X, Y, Z, levels=50, cmap='viridis')
ax2.plot(x_values_custom[:, 0], x_values_custom[:, 1], 'r-o')
ax2.scatter(x0[0], x0[1], color='blue', s=100, label='Initial Point')
ax2.set_xlabel('x0')
ax2.set_ylabel('x1')
ax2.set_title('Objective Function Contour and Optimization Path')
ax2.legend()

plt.show()

# Plotting objective function values
plt.figure(figsize=(8, 6))
plt.plot(f_values_custom, 'b-o')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Objective Function Values per Iteration')
plt.grid(True)
plt.show()
