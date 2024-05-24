import matplotlib.pyplot as plt
import numpy as np

from sr1 import sr1


def objective_function(x):
    return x[0] ** 2 + x[1] ** 2


def gradient(x):
    grad = np.zeros_like(x)
    grad[0] = 2 * (x[0] - 1)
    grad[1] = 2 * (x[1] - 2)
    return grad


# Initial guess
x0 = np.array([1.0, 1.0])

# SR1
optimal_solution, steps, x_values_custom, f_values_custom = sr1(objective_function, gradient, x0)
print("Optimal solution:", optimal_solution)
print("Objective function value at optimal solution:", objective_function(optimal_solution))
print("Number of steps taken:", steps)
print()

# Plotting
x_values_custom = np.array(x_values_custom)
f_values_custom = np.array(f_values_custom)

# Objective function surface
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
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
ax.set_title('Επιφάνεια Αντικειμενικής Συνάρτησης Quadratic Bowl \n και Διαδρομή Βελτιστοποίησης')
ax.legend()

# Contour plot
ax2 = fig.add_subplot(122)
ax2.contour(X, Y, Z, levels=50, cmap='viridis')
ax2.plot(x_values_custom[:, 0], x_values_custom[:, 1], 'r-o')
ax2.scatter(x0[0], x0[1], color='blue', s=100, label='Αρχικό Σημείο')
ax2.set_xlabel('x0')
ax2.set_ylabel('x1')
ax2.set_title('Contour Αντικειμενικής Συνάρτησης Quadratic Bowl \n και Διαδρομή Βελτιστοποίησης')
ax2.legend()

plt.show()

# Plotting objective function values
plt.figure(figsize=(8, 6))
plt.plot(f_values_custom, 'b-o')
plt.xlabel('Επανάληψη')
plt.ylabel('Τιμή Αντικειμεντικής Συνάρτησης')
plt.title('Τιμές Αντικειμεντικής Συνάρτησης Quadratic Bowl ανά επανάληψη k')
plt.grid(True)
plt.show()

