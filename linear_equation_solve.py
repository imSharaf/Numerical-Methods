import numpy as np

def gauss_elimination(A, steps=False):

    if isinstance(A, np.ndarray):
        pass
    else:
        A = np.array(A)

    A = A.astype(float)
    n = A.shape[0]

    if steps:
        print(f'Step 1:\n', A, '\n')

    for i in range(n):
    
        max_row = np.argmax(abs(A[i:, i])) + i
        A[[i, max_row]] = A[[max_row, i]]

        if A[i][i] == 0:
            raise ValueError("Matrix is singular.")

        if steps:
            print(f"Step {i+2}:\n", A, "\n")

        for j in range(i+1, n):
            factor = A[j][i] / A[i][i]
            A[j] = A[j] - factor * A[i]

    B = A[:,:-1]
    C = A[:,-1]
    D = np.linalg.solve(B,C)

    return D , A


def gauss_jordan(A, steps=False):

    if isinstance(A, np.ndarray):
        pass
    else:
        A = np.array(A)

    A = A.astype(float)
    n = A.shape[0]

    if steps:
        print(f'Step 1:\n', A, '\n')

    for i in range(n):

        max_row = np.argmax(abs(A[i:, i])) + i
        A[[i, max_row]] = A[[max_row, i]]

        if A[i, i] == 0:
            raise ValueError("Matrix is singular.")

        pivot = A[i, i]
        A[i] = A[i] / pivot

        for j in range(n):
            if j != i:
                factor = A[j, i]
                A[j] = A[j] - factor * A[i]

        if steps:
            print(f"Step {i+2}:\n", A, "\n")

    x = A[:, -1]

    return x, A

def gauss_seidel(A, iterations=10, x0=None, steps=False):

    if not isinstance(A, np.ndarray):
        A = np.array(A, dtype=float)
    else:
        A = A.astype(float)

    n = A.shape[0]

    coeff = A[:, :-1]
    b = A[:, -1]

    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float)

    for k in range(iterations):
        x_new = x.copy()

        for i in range(n):
            if coeff[i, i] == 0:
                raise ValueError("Zero diagonal element encountered.")

            sum1 = np.dot(coeff[i, :i], x_new[:i])
            sum2 = np.dot(coeff[i, i+1:], x[i+1:])

            x_new[i] = (b[i] - sum1 - sum2) / coeff[i, i]

        x = x_new

        if steps:
            print(f"Iteration {k+1}: {x}")

    return x, A
