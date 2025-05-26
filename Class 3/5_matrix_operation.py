import numpy as np

#defining a matrix using np.array() method
matrix_a = np.array([[5,6],[7,8]])
matrix_b = np.array([[1,2],[3,4]])


addition = np.add(matrix_a, matrix_b)
print ('Calculating Addition:\n', addition)
print('-----------------------')

subtraction = np.subtract(matrix_a, matrix_b)
print('Calculating Subtraction:\n', subtraction)
print('-----------------------')

multiplication = np.multiply(matrix_a, matrix_b)
print('Calculating Multiplication:\n', multiplication)
print('-----------------------')

division = np.divide(matrix_a, matrix_b)
print('Calculating Division:\n', division)
print('======================')

scalar_multiplication = 2 * matrix_a
print("Scalar Multiplication (2 * a):\n", scalar_multiplication)
print('-----------------------')

dot_product = np.dot(matrix_a, matrix_b)
print('Calculating Dot Product:\n', dot_product)
print('======================')