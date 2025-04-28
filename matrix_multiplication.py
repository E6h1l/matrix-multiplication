import numpy as np

def block_matrix_multiply(n, m, p, L, get_block):
    """
    Multiply matrices A and B using block approach.
    Matrices are divided into blocks of size LxL.
    
    Parameters:
    -----------
    n, m, p : int
        Dimensions of matrices (A is n×m, B is m×p)
    L : int
        Size of blocks for matrix division
    get_block : callable
        Function that returns a block of either A or B matrix
        Signature: get_block(matrix_name, row_start, row_end, col_start, col_end)
        matrix_name should be either 'A' or 'B'
        
    Returns:
    --------
    C : numpy.ndarray
        Result matrix C = A×B
    """
    # Initialize the result matrix
    C = np.zeros((n, p))
    
    # Iterate through blocks of C
    for i in range(0, n, L):
        i_end = min(i + L, n)
        for j in range(0, p, L):
            j_end = min(j + L, p)
            
            # Initialize block of C
            C_block = np.zeros((i_end - i, j_end - j))
            
            # Multiply the corresponding blocks from A and B
            for k in range(0, m, L):
                k_end = min(k + L, m)
                
                # Get blocks from A and B
                A_block = get_block('A', i, i_end, k, k_end)
                B_block = get_block('B', k, k_end, j, j_end)
                
                # Multiply and accumulate result
                C_block += A_block @ B_block
            
            # Set the result in C
            C[i:i_end, j:j_end] = C_block
    
    return C

# Example usage with blocks
def example():
    import time

    # Matrix dimensions
    n = 1024  # Rows A
    m = 1024  # Columns A | Rows B
    p = 1024  # Columns B

    # Block size L
    L = 64

    # Create full matrices for verification
    full_A = np.random.rand(n, m)
    full_B = np.random.rand(m, p)

    # Dictionaries to store blocks
    A_blocks = {}
    B_blocks = {}

    # Divide matrices into blocks of fixed size L
    print(f"Dividing matrices into blocks of size {L}×{L}...")
    for i in range(0, n, L):
        i_end = min(i + L, n)
        for j in range(0, m, L):
            j_end = min(j + L, m)
            # Blocks for A
            A_blocks[(i, i_end, j, j_end)] = full_A[i:i_end, j:j_end].copy()

    for i in range(0, m, L):
        i_end = min(i + L, m)
        for j in range(0, p, L):
            j_end = min(j + L, p)
            # Blocks for B
            B_blocks[(i, i_end, j, j_end)] = full_B[i:i_end, j:j_end].copy()

    # Function to get blocks
    def get_block(matrix_name, row_start, row_end, col_start, col_end):
        if matrix_name == 'A':
            key = (row_start, row_end, col_start, col_end)
            if key in A_blocks:
                return A_blocks[key]
            else:
                print(f"Warning: Block A[{row_start}:{row_end}, {col_start}:{col_end}] not found")
                return np.zeros((row_end - row_start, col_end - col_start))
        else:  # matrix_name == 'B'
            key = (row_start, row_end, col_start, col_end)
            if key in B_blocks:
                return B_blocks[key]
            else:
                print(f"Warning: Block B[{row_start}:{row_end}, {col_start}:{col_end}] not found")
                return np.zeros((row_end - row_start, col_end - col_start))

    # Multiplication using the block approach
    print(f"Multiplying matrices {n}×{m} and {m}×{p} using blocks {L}×{L}")
    start_time = time.time()
    C = block_matrix_multiply(n, m, p, L, get_block)
    block_time = time.time() - start_time
    print(f"Block multiplication time: {block_time:.4f} seconds")

    # Comparison with NumPy
    start_time = time.time()
    C_numpy = full_A @ full_B
    numpy_time = time.time() - start_time
    print(f"NumPy time: {numpy_time:.4f} seconds")

    # Check correctness
    diff = np.max(np.abs(C - C_numpy))
    print(f"Maximum difference: {diff}")

# Run the realistic example
if __name__ == "__main__":
    example()