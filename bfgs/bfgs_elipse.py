import matplotlib.pyplot as plt
import numpy as np

from bfgs import bfgs


def objective_function(x):
    return (x[0] - 1) ** 2 + (x[1] - 2) ** 2


def gradient(x):
    grad = np.zeros_like(x)
    grad[0] = 2 * (x[0] - 1)
    grad[1] = 2 * (x[1] - 2)
    return grad


# Initial guess
x0 = np.array([0.0, -2.0])

# BFGS
optimal_solution, steps, x_values_custom, f_values_custom = bfgs(objective_function, gradient, x0)
print("Βέλτιστο:", optimal_solution)
print("Τιμή αντικειμενικής συνάρτησης στο βέλτιστο:", objective_function(optimal_solution))
print("Επαναλήψεις που χρειάστηκαν:", steps)
print()

# Create a grid of points
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)
Z = objective_function([X, Y])

# Plotting the results
plt.figure(figsize=(12, 6))

# Contour plot of the objective function
plt.contour(X, Y, Z, levels=10, cmap='viridis')

# Plot initial point
plt.plot(x0[0], x0[1], 'ko', markersize=10, label='Αρχικό σημείο x_0')

# Custom BFGS Path
x_values_custom = np.array(x_values_custom)
plt.plot(x_values_custom[:, 0], x_values_custom[:, 1], 'r-o', label='BFGS')


plt.title('Μονοπάτι Βελτιστοποίησης και Contour της Αντικειμενικής Συνάρτησης Έλλειψης')
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.legend()
plt.show()


# Plotting objective function values
plt.figure(figsize=(8, 6))
plt.plot(f_values_custom, 'b-o')
plt.xlabel('Επανάληψη')
plt.ylabel('Τιμή Αντικειμεντικής Συνάρτησης')
plt.title('Τιμές Αντικειμεντικής Συνάρτησης Έλλειψης ανά επανάληψη k')
plt.legend()
plt.grid(True)
plt.show()