import numpy as np

# Creating a 2D array (3 rows, 4 columns)
matrix = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ]

# Accessing elements0
print(matrix[0][0])  # Output: 1 (first row, first column)
print(matrix[1][2])  # Output: 7 (second row, third column)
print(matrix[2][3])  # Output: 12 (third row, fourth column)

# Modifying elements
print('Before Modify: ',matrix)
matrix[1][1] = 100
print('After Modify: ',matrix)