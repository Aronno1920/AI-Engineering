import numpy as np

vector_a = np.array([7,8,8])
vector_b = np.array([2,3,4])

addition = np.add(vector_a, vector_b)
print ('Calculating Addition: ', addition)
print('-----------------------')

subtraction = np.subtract(vector_a, vector_b)
print('Calculating Subtraction: ', subtraction)
print('-----------------------')

multiplication = np.multiply(vector_a, vector_b)
print('Calculating Multiplication: ', multiplication)
print('-----------------------')

division = np.divide(vector_a, vector_b)
print('Calculating Division: ', division)
print('======================')

scalar_multiplication = 2 * vector_a
print("Scalar Multiplication (2 * a): ", scalar_multiplication)
print('-----------------------')

dot_product = np.dot(vector_a, vector_b)
print('Calculating Dot Product: ', dot_product)
print('-----------------------')

cross_product = np.cross(vector_a, vector_b)
print('Calculating Cross Product: ', cross_product)
print('======================')


