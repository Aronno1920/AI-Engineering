import numpy as np

# Example of vector
vector_1d = np.array([1, 2, 3])
vector_2d = np.array([[1, 2, 3], [3, 2, 3]])
vector_3d = np.array([[[1, 2, 3], [3, 2, 3]], [[1, 2, 3], [3, 2, 3]]])

print('Example of 1D Vector: ', vector_1d)
print('Shape:', vector_1d.shape, '| Dimension:', vector_1d.ndim)
print('-----------------------\n\n')

print('Example of 2D Vector (Matrix): ', vector_2d)
print('Shape:', vector_2d.shape, '| Dimension:', vector_2d.ndim)
print('-----------------------\n\n')

print('Example of 3D Vector (Tensor): ', vector_3d)
print('Shape:', vector_3d.shape, '| Dimension:', vector_3d.ndim)
print('-----------------------\n\n')
