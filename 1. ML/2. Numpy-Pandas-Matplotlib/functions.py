from typing import List

def sum_non_neg_diag(X: List[List[int]]) -> int:
    sum = 0
    flag = 0
    
    for i in range(min(len(X), len(X[0]))):
        if X[i][i] >= 0:
            sum += X[i][i]
            flag = 1
            
    if flag == 0:
        return -1
    
    return sum


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    if len(x) != len(y):
        return False
    
    y_copy = list(y.copy())
    
    for i in x:
        if i in y_copy:
            y_copy.remove(i)
        else:
            return False
        
    return True


def max_prod_mod_3(x: List[int]) -> int:
    max = -1
    
    for i in range(len(x) - 1):
        if x[i] % 3 == 0 or x[i + 1] % 3 == 0:
            cur = x[i] * x[i + 1]
            
            if cur > max:
                max = cur
                
    return max


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    res = [[0] * len(image[0]) for i in range(len(image))]
    
    for i in range(len(image)):
        for j in range(len(image[0])):
            for k in range(len(image[0][0])):
                res[i][j] += image[i][j][k] * weights[k]
                
    return res


def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    res = 0
    x_copy = []
    y_copy = []
    
    for i in range(len(x)):
        for j in range(x[i][1]):
            x_copy.append(x[i][0])
            
    for i in range(len(y)):
        for j in range(y[i][1]):
            y_copy.append(y[i][0])
            
    if len(x_copy) != len(y_copy):
        return -1
    
    for i in range(len(x_copy)):
        res += x_copy[i] * y_copy[i]
        
    return res


def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    n = len(X)
    m = len(Y)
    d = len(X[0])
    
    if d != len(Y[0]):
        return -1
    
    res = [[0] * m for i in range(n)]
    
    for i in range(n):
        for j in range(m):
                scr = 0
                norm_x = 0
                norm_y = 0
                
                for k in range(d):
                    scr += X[i][k] * Y[j][k]
                    norm_x += X[i][k] * X[i][k]
                    norm_y += Y[j][k] * Y[j][k]
                    
                norm_x = norm_x ** 0.5
                norm_y = norm_y ** 0.5
                
                if norm_x == 0 or norm_y == 0:
                    res[i][j] = 1
                else:
                    res[i][j] = scr / (norm_x * norm_y)
                
    return res