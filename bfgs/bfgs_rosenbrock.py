import numpy as np
# from scipy.optimize import line_search
from scipy.optimize.linesearch import line_search_wolfe2
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve

def objective_function(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (x[0] - 1) ** 2


def gradient(x):
    grad = np.zeros_like(x)
    grad[0] = -400 * x[0] * (x[1] - x[0] ** 2) + 2 * (x[0] - 1)
    grad[1] = 200 * (x[1] - x[0] ** 2)
    return grad


def bfgs(f, grad, x0, max_iter=100, tol=1e-5, epsilon=1e-8):
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
        # p_k = -np.dot(B_k, grad_k)  # Search direction
        # p_k = -np.linalg.inv(B_k) @ grad_k

        # Perform Cholesky decomposition of B_k
        try:
            L, lower = cho_factor(B_k)
            p_k = cho_solve((L, lower), -grad_k)
            # print(p_k)
        except np.linalg.LinAlgError:
            print("Cholesky decomposition failed. Using gradient descent direction.")
            p_k = -grad_k

        # Line search method to find step size α
        alpha_k = line_search_wolfe2(f, grad, x_k, p_k)[0]
        if alpha_k is None:
            alpha_k = 1e-4  # Default to a smaller step size if line search fails

        # Update x_k
        x_k1 = x_k + alpha_k * p_k

        # Compute s_k and y_k
        s_k = x_k1 - x_k
        y_k = grad(x_k1) - grad_k

        # # Avoid division by zero and numerical instability
        # if np.dot(s_k, y_k) > 1e-10:
        #     rho_k = 1.0 / np.dot(y_k, s_k)
        #     V_k = np.eye(n) - rho_k * np.outer(s_k, y_k)
        #     B_k = V_k.T @ B_k @ V_k + rho_k * np.outer(s_k, s_k)

        s_k = s_k.reshape(-1, 1)
        y_k = y_k.reshape(-1, 1)

        Bksk = B_k @ s_k
        skT_Bk_sk = s_k.T @ B_k @ s_k
        ykT_sk = y_k.T @ s_k

        # Add conditions to prevent division by zero or small values
        if skT_Bk_sk < epsilon:
            skT_Bk_sk = epsilon
        if ykT_sk < epsilon:
            ykT_sk = epsilon

        term1 = (Bksk @ Bksk.T) / skT_Bk_sk
        term2 = (y_k @ y_k.T) / ykT_sk

        B_k = B_k - term1 + term2


        # Update x_k for the next iteration
        x_k = x_k1

        steps += 1
        x_values.append(x_k.copy())
        f_values.append(f(x_k))
        print(f"Step {steps}: x = {x_k}, f(x) = {f(x_k)}")
    return x_k, steps, x_values, f_values


# Initial guess
x0 = np.array([4.0, 4.0])

# Custom BFGS
optimal_solution, steps, x_values_custom, f_values_custom = bfgs(objective_function, gradient, x0)
print("Optimal solution:", optimal_solution)
print("Objective function value at optimal solution:", objective_function(optimal_solution))
print("Number of steps taken:", steps)

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
ax.scatter(x0[0], x0[1], objective_function(x0), color='blue', s=100, label='Αρχικό Σημείο')
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('f(x)')
ax.set_title('Επιφάνεια Αντικειμενικής Συνάρτησης Rosenbrock \n και Διαδρομή Βελτιστοποίησης')
ax.legend()

# Contour plot
ax2 = fig.add_subplot(122)
ax2.contour(X, Y, Z, levels=50, cmap='viridis')
ax2.plot(x_values_custom[:, 0], x_values_custom[:, 1], 'r-o')
ax2.scatter(x0[0], x0[1], color='blue', s=100, label='Αρχικό Σημείο')
ax2.set_xlabel('x0')
ax2.set_ylabel('x1')
ax2.set_title('Contour Αντικειμενικής Συνάρτησης Rosenbrock \n και Διαδρομή Βελτιστοποίησης')
ax2.legend()

plt.show()

# Plotting objective function values
plt.figure(figsize=(8, 6))
plt.plot(f_values_custom, 'b-o')
plt.xlabel('Επανάληψη')
plt.ylabel('Τιμή Αντικειμεντικής Συνάρτησης')
plt.title('Τιμές Αντικειμεντικής Συνάρτησης Rosenbrock ανά επανάληψη k')
plt.grid(True)
plt.show()
