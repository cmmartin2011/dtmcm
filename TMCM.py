import time
import numpy as np
from DPCA import *
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import matplotlib.lines as mlines
import xlwings as xw

def ToExcel(matrix, names):
    wb = xw.Book()
    sht = wb.sheets[0]
    n_cols = matrix.shape[1]
    headers = [f"COMP{i+1}" for i in range(n_cols)]
    sht.range(1, 2).value = headers
    for i, name in enumerate(names):
        sht.range(i+2, 1).value = name
    sht.range(2, 2).value = matrix

def SVD(matrix, n):
    rows, cols = matrix.shape
    if n <= min(rows, cols):
        U, _, _ = np.linalg.svd(matrix, full_matrices = False)
    else:
        U, _, _ = np.linalg.svd(matrix, full_matrices = True)
    return U[:, :n]

def tmcmALS(X, P, Q, ALSMaxIter):
    start_time = time.time()
    A = SVD(X, P)
    for k in range(ALSMaxIter):
        B = SVD(X.T @ A, Q)
        A = SVD(X @ B, P)
    G = A.T @ X @ B
    Fit = np.linalg.norm(A @ G @ B.T, 'fro')**2 / np.linalg.norm(X, 'fro')**2
    elapsed_time = time.time() - start_time
    return A, B, G, Fit, elapsed_time

def convex_hull(X, Pmax, Qmax, ALSMaxIter = 100):
    results = []
    for P in range(2, Pmax + 1):
        for Q in range(2, Qmax + 1):
            _, _, _, Fit, _ = tmcmALS(X, P, Q, ALSMaxIter)
            Fit_percent = round(Fit * 100, 2)
            results.append([P, Q, Fit_percent])
    return np.array(results)

def plot_ch(models_matrix):
    P = models_matrix[:,0]
    Q = models_matrix[:,1]
    Fit = models_matrix[:,2]
    X = P + Q
    Y = Fit
    points = np.column_stack((X, Y))
    hull = ConvexHull(points)
    plt.figure(figsize=(8,6))
    hull_indices = set(hull.vertices)
    all_indices = set(range(len(points)))
    interior_indices = all_indices - hull_indices
    plt.scatter(points[list(interior_indices),0], points[list(interior_indices),1],
                color = 'lightgray', s = 30, zorder = 2, label = 'Interior models')
    plt.scatter(points[list(hull_indices),0], points[list(hull_indices),1],
                color = 'lightcoral', s = 30, zorder = 3, label = 'Hull models')
    for simplex in hull.simplices:
        plt.plot(points[simplex,0], points[simplex,1],
                 color = 'mediumaquamarine', linewidth = 1.2, zorder = 1)
    for i in hull.vertices:
        x, y = points[i]
        plt.text(x, y + 0.5, str(i + 1), color = 'blue', fontsize = 8, ha = 'center', zorder = 4)
    plt.xlabel("Sum of number of components")
    plt.ylabel("Fit(%)")
    gray_point = plt.Line2D([], [], color = 'lightgray', marker = 'o', linestyle = 'None',
                            markersize = 6, label = 'Interior models')
    red_point = plt.Line2D([], [], color = 'lightcoral', marker = 'o', linestyle = 'None',
                           markersize = 6, label = 'Hull models')
    green_line = mlines.Line2D([], [], color = 'mediumaquamarine', linewidth = 1.2, label = 'Convex hull')
    plt.legend(handles = [gray_point, red_point, green_line], loc = 'upper left')
    plt.show()

def tmcmDisjoint(X, P, Q, ALSMaxIter, Tol = 1e-5):
    start_time = time.time()
    A = SVD(X, P)
    for k in range(ALSMaxIter):
        B = SVD(X.T @ A, Q)
        A = SVD(X @ B, P)
    Phi = A; Psi = B
    _, _, _, A, _ = VS(Psi.T @ X.T, P, Tol)
    _, _, _, B, _ = VS(Phi.T @ X, Q, Tol)
    G = A.T @ X @ B
    Fit = np.linalg.norm(A @ G @ B.T, 'fro')**2 / np.linalg.norm(X, 'fro')**2
    elapsed_time = time.time() - start_time
    return A, B, G, Fit, elapsed_time

rows_names, columns_names, datamatrix = get_datamatrix_csv2(r"BaseBE.csv", center = True, scale = True)
models = convex_hull(datamatrix, 10, 7, ALSMaxIter = 100)
plot_ch(models)
ToExcel(models, ['Models'])

bestA, bestB, bestG = None, None, None
best = 0;
for i in range(20):
    A, B, G, Fit, elapsed_time = tmcmDisjoint(datamatrix, 5, 5, 100, 1e-5)
    print(i + 1, " - ", Fit, elapsed_time)
    if (Fit > best):
        best = Fit
        bestA, bestB, bestG = A, B, G
print("Fit:", best)
ToExcel(bestA, rows_names)
ToExcel(bestB, columns_names)
ToExcel(bestG, ['DC1-A', 'DC2-A', 'DC3-A', 'DC4-A', 'DC5-A'])
