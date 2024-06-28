import numpy as np

def scale_array_from_A_to_B(array, A, B):
    """
    Scale array from range A to range B

    Args
        array: (B, D) array
        A: (2, ) array [min, max] where min, max is (1, D) array or scalar
        B: (2, ) array [min, max] where min, max is (1, D) array or scalar
    """
    assert np.all(A[0] <= A[1]) and np.all(B[0] <= B[1]) # min, max is in correct order
    assert np.all(array >= A[0]) and np.all(array <= A[1]) # array is within range A

    return (B[1] - B[0]) * (array - A[0]) / (A[1] - A[0]) + B[0]