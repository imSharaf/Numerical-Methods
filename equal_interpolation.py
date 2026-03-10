import numpy as np 
import math 

def divided_diff(array):
    
    list = [array.tolist()]

    for i in range(len(array)-1):
        arr = []
        for j in range(len(list[-1])-1):
            arr.append(list[-1][j+1] - list[-1][j])
        list.append(arr)

    return list

def falling_factorial(p, k):
    result = 1
    for i in range(k):
        result *= (p - i)
    return result

def rising_factorial(p, k):
    result = 1
    for i in range(k):
        result *= (p + i)
    return result

def central_sequence_down(p, n):
    if n == 0:
        return 1
    if n == 1:
        return p

    term = p
    left = 1
    right = 1

    for i in range(2, n+1):
        if i % 2 == 0:
            # multiply by (p - left)
            term *= (p - left)
            left += 1
        else:
            # multiply by (p + right)
            term *= (p + right)
            right += 1

    return term

def central_sequence_up(p, n):
    if n == 0:
        return 1
    if n == 1:
        return p

    term = p
    left = 1   
    right = 1  

    for i in range(2, n+1):
        if i % 2 == 0:
            term *= (p + right)
            right += 1
        else:
            term *= (p - left)
            left += 1

    return term

def stirling_p_term(p, i):
    
    if i == 0:
        return 1
    elif i == 1:
        return p
    elif i == 2:
        return p**2
    
    if i % 2 == 1:  
        term = p
        count = (i - 1) // 2
    else:           
        term = p**2
        count = (i - 2) // 2
    
    for k in range(1, count + 1):
        term *= (p**2 - k**2)
    
    return term

def is_arithmetic(seq):
    if len(seq) < 2:
        return True
    
    d = seq[1] - seq[0]
    
    for i in range(2, len(seq)):
        if seq[i] - seq[i-1] != d:
            return False
    
    return True

def gauss_forward(table, value, central=False):
    if isinstance(table, np.ndarray):
        pass
    else:
        table = np.array(table)

    if table.shape[1] != 2:
        raise ValueError('The dimension should be n x 2')
    
    if np.isnan(table).any() == True:
        raise ValueError('There are Nan values in the data')

    if is_arithmetic(table[:,0]) != True:
        raise ValueError('The difference are not equal of the provided data')
    
    if central == True:
        if table.shape[0] % 2 != 0:
            n = table.shape[0] // 2
            p = (value - table[n][0]) / abs(table[0][0] - table[1][0])
        else:
            n = (table.shape[0] // 2) - 1
            p = (value - table[n][0]) / abs(table[0][0] - table[1][0])
    else: 
        p = (value - table[0][0]) / abs(table[0][0] - table[1][0])

    if central == True:
        factor = [central_sequence_down(p,i) for i in range(table.shape[0])]
    else:
        factor = [falling_factorial(p,i) for i in range(table.shape[0])]

    div_diff = divided_diff(table[:,1])
    
    if central == True:
        diff_element = []
        for i in range(table.shape[0]):
            if len(div_diff[i]) % 2 != 0:
                diff_element.append(div_diff[i][len(div_diff[i]) // 2])
            else:
                diff_element.append((div_diff[i][(len(div_diff[i]) // 2) - 1]))
    else:
        diff_element = [div_diff[i][0] for i in range(table.shape[0])]

    sum = 0

    for i in range(table.shape[0]):
        sum += (factor[i] * diff_element[i]) / math.factorial(i)

    return sum

def gauss_backward(table, value, central=False):
    if isinstance(table, np.ndarray):
        pass
    else:
        table = np.array(table)

    if table.shape[1] != 2:
        raise ValueError('The dimension should be n x 2')
    
    if np.isnan(table).any() == True:
        raise ValueError('There are Nan values in the data')

    if is_arithmetic(table[:,0]) != True:
        raise ValueError('The difference are not equal of the provided data')
    
    if central == True:
        n = (table.shape[0] // 2)
        p = (value - table[n][0]) / abs(table[0][0] - table[1][0])
    else:
        p = (value - table[-1][0]) / abs(table[0][0] - table[1][0])

    if central == True:
        factor = [central_sequence_up(p,i) for i in range(table.shape[0])]
    else:
        factor = [rising_factorial(p,i) for i in range(table.shape[0])]

    div_diff = divided_diff(table[:,1])
    
    if central == True:
        diff_element = []
        for i in range(table.shape[0]):
            diff_element.append(div_diff[i][len(div_diff[i]) // 2])
    else:
        diff_element = [div_diff[i][-1] for i in range(table.shape[0])]

    sum = 0

    for i in range(table.shape[0]):
        sum += (factor[i] * diff_element[i]) / math.factorial(i)

    return sum

def stirling_central(table, value):
    if isinstance(table, np.ndarray):
        pass
    else:
        table = np.array(table)

    if table.shape[1] != 2:
        raise ValueError('The dimension should be n x 2')
    
    if np.isnan(table).any() == True:
        raise ValueError('There are Nan values in the data')

    if is_arithmetic(table[:,0]) != True:
        raise ValueError('The difference are not equal of the provided data')
    
    if table.shape[0] % 2 == 0:
        n = (table.shape[0] // 2) - 1
        p = (value - table[n][0]) / abs(table[0][0] - table[1][0])
    else:
        n = (table.shape[0] // 2)
        p = (value - table[n][0]) / abs(table[0][0] - table[1][0])

    factor = [stirling_p_term(p,i) for i in range(table.shape[0])]

    div_diff = divided_diff(table[:,1])
    diff_element = []

    x = div_diff

    if len(x) % 2 != 0:
        for i in range(len(x)):
            if i % 2 == 0:
                diff_element.append(x[i][len(x[i]) // 2])
            else:
                diff_element.append(((x[i][len(x[i]) // 2]) + (x[i][(len(x[i]) // 2) - 1])) / 2)

    else:
        for i in range(len(x)):
            if i % 2 == 0:
                diff_element.append(x[i][(len(x[i]) // 2) - 1])
            else:
                diff_element.append(((x[i][len(x[i]) // 2]) + (x[i][(len(x[i]) // 2) - 1])) / 2)

    sum = 0

    for i in range(table.shape[0]):
        sum += (factor[i] * diff_element[i]) / math.factorial(i)

    return sum