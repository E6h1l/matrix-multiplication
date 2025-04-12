import numpy as np

def matrix_multiply(A, B, L):
    """
    Multiply matrices A and B using divide-and-conquer approach.
    Only perform direct multiplication if dimensions are <= L.
    
    Parameters:
    -----------
    A, B : numpy.ndarray
        Input matrices to multiply
    L : int
        Threshold for direct multiplication
        
    Returns:
    --------
    C : numpy.ndarray
        Result matrix C = A×B
    """
    # Get dimensions
    n, m = A.shape
    m2, p = B.shape
    
    # Check if matrices can be multiplied
    if m != m2:
        raise ValueError("Matrix dimensions incompatible for multiplication")
    
    # Base case: if matrices are small enough, use standard multiplication
    if n <= L and m <= L and p <= L:
        return A @ B
    
    # Special case: if any dimension is 1, use standard multiplication
    if n == 1 or m == 1 or p == 1:
        return A @ B
    
    # Divide matrices into blocks
    n_half = n // 2
    m_half = m // 2
    p_half = p // 2
    
    # Split A
    A11 = A[:n_half, :m_half]
    A12 = A[:n_half, m_half:]
    A21 = A[n_half:, :m_half]
    A22 = A[n_half:, m_half:]
    
    # Split B
    B11 = B[:m_half, :p_half]
    B12 = B[:m_half, p_half:]
    B21 = B[m_half:, :p_half]
    B22 = B[m_half:, p_half:]
    
    # Recursive multiplication of blocks
    C11 = matrix_multiply(A11, B11, L) + matrix_multiply(A12, B21, L)
    C12 = matrix_multiply(A11, B12, L) + matrix_multiply(A12, B22, L)
    C21 = matrix_multiply(A21, B11, L) + matrix_multiply(A22, B21, L)
    C22 = matrix_multiply(A21, B12, L) + matrix_multiply(A22, B22, L)
    
    # Combine blocks to form C
    top = np.hstack((C11, C12))
    bottom = np.hstack((C21, C22))
    C = np.vstack((top, bottom))
    
    return C

# Example usage
if __name__ == "__main__":
    # Set the threshold L
    L = 64
    
    n = 256
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    
    print(f"Multiplying {n}x{n} matrices with threshold L={L}")
    print(f"Matrix A: {A}")
    print(f"Matrix B: {B}")
    
    # Compute C using divide-and-conquer approach
    import time
    start_time = time.time()
    C = matrix_multiply(A, B, L)
    dc_time = time.time() - start_time
    print(f"Divide-and-conquer time: {dc_time:.4f} seconds")
    print(f"Matric C: {C}")

    # Compare with NumPy's built-in multiplication
    start_time = time.time()
    C_numpy = A @ B
    numpy_time = time.time() - start_time
    print(f"NumPy time: {numpy_time:.4f} seconds")
    
    # Verify correctness
    diff = np.max(np.abs(C - C_numpy))
    print(f"Maximum difference: {diff}")