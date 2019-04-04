import numpy as np

BS_MATRIX = 1 / np.sqrt(2) * np.array([[1 + 0j, 0 + 1j], [0 + 1j, 1 + 0j]], dtype = np.complex128)

CPHASE = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, -1]], dtype = np.complex128)

CPHASE_MOD = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]], dtype = np.complex128)

IDENTITY = np.eye(2, dtype = np.complex128)
