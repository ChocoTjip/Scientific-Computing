import numpy as np
from scipy.sparse import csr_matrix
from numpy.linalg import norm

def ssor_iteration(A, u, f, omega, N):
    """
    Perform a single SSOR iteration on the linear system A * u = f.

    Args:
        A (csr_matrix): Sparse matrix (N+1)^2 x (N+1)^2.
        u (ndarray): Current solution vector of size (N+1)^2.
        f (ndarray): Right-hand side vector of size (N+1)^2.
        omega (float): Relaxation parameter (0 < omega < 2).
        N (int): The size parameter defining the structure of A.

    Returns:
        u (ndarray): Updated solution vector after one SSOR iteration.
    """
    num_rows = A.shape[0]
    u_new = u.copy()  # To hold the updated solution
   
    # Forward sweep
    for i in range(num_rows):
        aux = u[i]
       
        # Compute sums for off-diagonal elements
        start_idx = max(0, i - N)
        end_idx = min(num_rows, i + N + 1)
       
        sum1 = A[i, start_idx:i].dot(u[start_idx:i])  # Sum over the lower triangular part
        sum2 = A[i, i + 1:end_idx].dot(u[i + 1:end_idx])  # Sum over the upper triangular part
       
        # Update u[i]
        u_new[i] = (f[i] - sum1 - sum2) / A[i, i]
        u_new[i] = (1 - omega) * aux + omega * u_new[i]
   
    # Backward sweep
    for i in range(num_rows - 1, -1, -1):
        aux = u_new[i]
       
        # Compute sums for off-diagonal elements
        start_idx = max(0, i - N)
        end_idx = min(num_rows, i + N + 1)
       
        sum1 = A[i, start_idx:i].dot(u_new[start_idx:i])  # Sum over the lower triangular part
        sum2 = A[i, i + 1:end_idx].dot(u_new[i + 1:end_idx])  # Sum over the upper triangular part
       
        # Update u[i]
        u_new[i] = (f[i] - sum1 - sum2) / A[i, i]
        u_new[i] = (1 - omega) * aux + omega * u_new[i]
   
    return u_new

def ssor_until_convergence(A, u, f, omega, N, tol, max_iter=1000):
    """
    Perform SSOR iterations until convergence with a given tolerance.
   
    Args:
        A (csr_matrix): Sparse matrix (N+1)^2 x (N+1)^2.
        u (ndarray): Initial solution vector of size (N+1)^2.
        f (ndarray): Right-hand side vector of size (N+1)^2.
        omega (float): Relaxation parameter (0 < omega < 2).
        N (int): The size parameter defining the structure of A.
        tol (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.

    Returns:
        u (ndarray): Solution vector after convergence.
        iter_count (int): Number of iterations performed.
    """
    iter_count = 0
    while iter_count < max_iter:
        u_new = ssor_iteration(A, u, f, omega, N)
       
        # Compute the norm of the difference between the new and old solutions
        diff = norm(u_new - u, ord=np.inf)  # Use infinity norm for convergence criterion
       
        # Check if the difference is less than the tolerance
        if diff < tol:
            break
       
        u = u_new
        iter_count += 1
   
    return u_new, iter_count

# Example usage
N = 2
size = (N + 1) ** 2
A = csr_matrix(np.array([[4, -1, 0, -1, 0, 0],
                         [-1, 4, -1, 0, -1, 0],
                         [0, -1, 4, 0, 0, -1],
                         [-1, 0, 0, 4, -1, 0],
                         [0, -1, 0, -1, 4, -1],
                         [0, 0, -1, 0, -1, 4]]))
u = np.zeros(size)  # Initial guess
f = np.ones(size)  # Right-hand side
omega = 1.5  # Relaxation factor
tol = 1e-6  # Convergence tolerance

# Perform SSOR iterations until convergence
u_final, iterations = ssor_until_convergence(A, u, f, omega, N, tol)

print("Solution vector after convergence:", u_final)
print("Number of iterations:", iterations)