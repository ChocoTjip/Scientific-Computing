import numpy as np
from openpyxl import Workbook

h=.2

N = int(1/h+1)
Ah_oneD = np.zeros([N,N])

Ah_oneD[0,0], Ah_oneD[-1,-1] = 2*[h**2]

for i in range(1,N-1):
    Ah_oneD[i,i] = 2

for i in range(2,N-1):
    Ah_oneD[i-1,i] = -1
    Ah_oneD[i,i-1] = -1


# Ah_twoD = np.tensordot(Ah_oneD, Ah_oneD)


# Thath = np.zeros([N, N])
# Ihath = np.identity(N)

# Thath[0,0], Thath[-1,-1] = 2*[h**2]
# Thath[1,1],Thath[-2,-2] = 2*[4]
# Thath[1,2],Thath[-2,-3] = 2*[-1]
# for i in range(2, N-2):
#     Thath[i,i] = 4
#     Thath[i,i-1] = -1
#     Thath[i,i+1] = -1

# Ihath[0,0], Ihath[-1,-1] = 2*[0]

# for i in range(1,N-1):
#     Ah[N*i:N*(i+1),N*i:N*(i+1)] = Thath

# for i in range(2,N-2):
#     Ah[N*i:N*(i+1),N*(i+1):N*(i+2)] = -Ihath
#     Ah[N*i:N*(i+1),N*(i-1):N*i] = -Ihath

# Ah[N:2*N,2*N:3*N] = -Ihath
# Ah[N:2*N,2*N:3*N] = -Ihath

# Ah[0:N,0:N] = h**2*np.identity(N)
# Ah[-N:,-N:] = h**2*np.identity(N)

# Ah = h**-2*Ah

def row_col(index):
    row = index // N
    col = index % N
    dim = index % N**2
    return row, col, dim

# for i in range(N**2):
#     row, col = row_col(i)
#     if row == 0 or row == N-1 or col == 0 or col == N-1:
#         Ah[i,i] = 1
#     else if row == 1 or row == N-2
#     else:
#         Ah[i,i] = 4/h**2
#         Ah[i,i-1] = -1/h**2
#         Ah[i,i+1] = -1/h**2
#         Ah[i,i+N] = -1/h**2
#         Ah[i,i-N] = -1/h**2


def write_matrix_to_excel(matrix, filename="matrix.xlsx"):
    """
    Write a matrix to an Excel file.
    
    :param matrix: List of lists representing the matrix (2D array).
    :param filename: The name of the Excel file to write to.
    """
    # Create a new Excel workbook and select the active sheet
    workbook = Workbook()
    sheet = workbook.active
    
    # Write the matrix to the Excel sheet
    for index, value in np.ndenumerate(matrix):  # Excel rows start at 1
        sheet.cell(row=index[0]+1, column=index[1]+1, value=value)
    
    # Save the workbook to the specified file
    workbook.save(filename)
    print(f"Matrix written to {filename}")

I = np.eye(N)
Ah_twoD = np.kron(Ah_oneD, I)+np.kron(I, Ah_oneD)
Ah_threeD = np.kron(np.kron(Ah_oneD, I), I)+np.kron(np.kron(I, Ah_oneD), I)+np.kron(np.kron(I, I), Ah_oneD)

for i in range(N**2):
    if Ah_twoD[i,i] != 4:
        Ah_twoD[i] = np.zeros(N**2)
        Ah_twoD[i, i] = h**2

for i in range(N**3):
    if Ah_threeD[i,i] != 6:
        Ah_threeD[i] = np.zeros(N**3)
        Ah_threeD[i, i] = h**2


write_matrix_to_excel(Ah_threeD)
    