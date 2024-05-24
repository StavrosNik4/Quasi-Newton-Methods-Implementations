import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import line_search


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
            # print(p_k)
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
