import numpy as np


@profile
def test_function() -> np.ndarray:
    arr = np.random.rand(1_000, 1_000)
    arr_inv = np.linalg.inv(arr)
    arr_inv_squared = arr_inv**2

    return arr_inv_squared


if __name__ == "__main__":

    test_function()
