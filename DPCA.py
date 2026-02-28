import numpy as np
import pandas as pd
import time
from scipy.sparse.linalg import svds

# This function allows reading data from a CSV file.
def get_datamatrix_csv2(path_file, center=True, scale=False):
    file = pd.read_csv(path_file, sep=';', index_col=0)
    rows_names = file.index.tolist()
    columns_names = file.columns.to_list()
    datamatrix = file.to_numpy(dtype=float)
    if center:
        datamatrix -= np.mean(datamatrix, axis=0)
    if scale:
        std_dev = np.std(datamatrix, axis=0, ddof=1)
        std_dev[std_dev == 0] = 1
        datamatrix /= std_dev
    return rows_names, columns_names, datamatrix

def generate_binarymatrix(p, q):
    if p <= q:
        raise ValueError("The number (p) of rows must be strictly greater than the number (q) of columns.")
    while True:
        columns = np.random.permutation(q).tolist()
        while len(columns) < p:
            columns += np.random.permutation(q).tolist()
        binarymatrix = np.zeros((p, q), dtype=int)
        for i in range(p):
            binarymatrix[i, columns[i]] = 1
        if not has_zero_column(binarymatrix):
            return binarymatrix

def get_submatrix(datamatrix, binarymatrix, col):
    pos = np.where(binarymatrix[:, col] == 1)[0]
    submatrix = datamatrix[:, pos]
    return submatrix

def get_rightsingularvector(submatrix):
    _, _, v = svds(submatrix, k=1)
    return v[0, :]

def get_loadingmatrix(datamatrix, binarymatrix):
    p, q = binarymatrix.shape
    loadingmatrix = np.zeros((p, q))
    for col in range(q):
        submatrix = get_submatrix(datamatrix, binarymatrix, col)
        pos = np.where(binarymatrix[:, col] == 1)[0]
        if submatrix.shape[1] == 1:
            loadingmatrix[:, col] = binarymatrix[:, col]
        else:
            right_singular_vector = get_rightsingularvector(submatrix)
            loadingmatrix[pos, col] = right_singular_vector
    return loadingmatrix

def get_fit(datamatrix, loadingmatrix):
    scorematrix = np.dot(datamatrix, loadingmatrix)
    aprox = np.dot(scorematrix, loadingmatrix.T)
    frobenius_norm_sq = np.linalg.norm(datamatrix, 'fro') ** 2
    error_sum_sq = np.sum((datamatrix - aprox) ** 2)
    return error_sum_sq / frobenius_norm_sq

def has_zero_column(binarymatrix):
    for col in range(binarymatrix.shape[1]):
        if np.all(binarymatrix[:, col] == 0):
            return True
    return False

def get_var(datamatrix, loadingmatrix):
    num_objects = datamatrix.shape[0]
    mean_variables = np.mean(datamatrix, axis=0)
    var_data = np.mean(np.sum((datamatrix - mean_variables) ** 2, axis=1))
    A = datamatrix @ loadingmatrix
    mean_components = np.mean(A, axis=0)
    var_components = np.sum((A - mean_components) ** 2, axis=0) / num_objects
    var = np.round((var_components / var_data) * 100, 2)
    return var

def sort_columns(loadingmatrix, var):
    sorted_indices = np.argsort(var)[::-1]
    sorted_loadingmatrix = loadingmatrix[:, sorted_indices]
    sorted_var = var[sorted_indices]
    return sorted_loadingmatrix, sorted_var

# This is the function that calculates disjoint components in data matrices.
def VS(datamatrix, q, Tol):
    start_time = time.time()
    p = datamatrix.shape[1]
    binarymatrix = generate_binarymatrix(p, q)
    prev_fit = float('inf')
    convergence = False
    
    iter = 0
    while not convergence:
        iter += 1
        for row in range(p):
            best_col = None
            best_fit = float('inf')
            for col in range(q):
                temp_binarymatrix = binarymatrix.copy()
                temp_binarymatrix[row, :] = 0
                temp_binarymatrix[row, col] = 1
                if has_zero_column(temp_binarymatrix):  
                    fit = float('inf')
                else:
                    loadingmatrix = get_loadingmatrix(datamatrix, temp_binarymatrix)
                    fit = get_fit(datamatrix, loadingmatrix)
                if fit < best_fit:
                    best_fit = fit
                    best_col = col
            binarymatrix[row, :] = 0
            binarymatrix[row, best_col] = 1
        loadingmatrix = get_loadingmatrix(datamatrix, binarymatrix)
        current_fit = get_fit(datamatrix, loadingmatrix)
        if abs(prev_fit - current_fit) < Tol:
            convergence = True
        prev_fit = current_fit

    execution_time = round(time.time() - start_time, 2)
    var = get_var(datamatrix, loadingmatrix)
    loadingmatrix, var = sort_columns(loadingmatrix, var)
    return iter, execution_time, current_fit, loadingmatrix, var

def generate_disjoint_loading_matrix(p, q):
    if p <= q:
        raise ValueError("The number (p) of rows must be strictly greater than the number (q) of columns.")
    matrix = np.zeros((p, q), dtype=float)
    while True:
        matrix.fill(0)
        for row in range(p):
            random_col = np.random.randint(0, q)
            matrix[row, random_col] = np.random.uniform(-1, 1)
        if not has_zero_column(matrix):
            break

    norms = np.linalg.norm(matrix, axis=0)
    norms[norms == 0] = 1
    normalized_matrix = matrix / norms
    return normalized_matrix

# This function generates data matrices with a disjoint structure for conducting simulations.
def generate_data_matrix(n, p, q):
    B = generate_disjoint_loading_matrix(p, q)
    A = np.random.uniform(-1, 1, (n, q))
    datamatrix = A @ B.T
    return datamatrix

