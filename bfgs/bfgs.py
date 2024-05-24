import numpy as np
from scipy.optimize.linesearch import line_search_wolfe2

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
            print()
            break

        # Perform Cholesky decomposition of B_k
        try:

            # Perform Cholesky decomposition
            L = np.linalg.cholesky(B_k)

            # Define L transpose
            L_T = L.T

            # Solve Ly = -gradient using forward substitution
            y = np.linalg.solve(L, -grad_k)

            # Solve L_T * p = y using backward substitution
            p_k = np.linalg.solve(L_T, y)


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