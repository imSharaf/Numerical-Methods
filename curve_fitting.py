#Dependencies

import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
import seaborn as sns 
import math

#evaluation metrics

def r2(y,y_pred):
    r2 = 1 - ((y - np.array(y_pred))**2).sum() / ((y - np.array(y_pred).mean())**2).sum()

    return r2

def mse(y,y_pred):
    mse = ((y - np.array(y_pred))**2).sum() / n

    return mse

# linear fit 

def linear_fit(x,y,plot=False):

    if len(x) != len(y):
        raise ValueError("Dependent and Independent variables dimension mismatch")
    
    if isinstance(x, np.ndarray):
        pass
    else:
        x = np.array(x)

    if isinstance(y, np.ndarray):
        pass
    else:
        y = np.array(y)

    sum_x = x.sum()
    sum_y = y.sum()
    n = x.shape[0]
    sum_xy = (x*y).sum()
    sum_xx = (x*x).sum()

    a = (n* sum_xy - sum_x*sum_y)/(n*sum_xx - (sum_x)**2)
    b = (sum_y - a*sum_x)/n 

    y_pred = []

    for i in x:
        y_pred.append(a*i+b)

    if plot:
        sns.scatterplot(x=x, y=y, label='True')
        sns.lineplot(x=x, y=y_pred, color='red', label='Best Fit')
        plt.legend()
        plt.grid()
        plt.show()

    return [y_pred, a,b]

#parabolic fit

def parabolic_fit(x,y,plot=False):

    if len(x) != len(y):
        raise ValueError("Dependent and Independent variables dimension mismatch")
    
    if isinstance(x, np.ndarray):
        pass
    else:
        x = np.array(x)

    if isinstance(y, np.ndarray):
        pass
    else:
        y = np.array(y)

    sum_y = y.sum()
    sum_x = x.sum()
    sum_xx = (x*x).sum()
    n = len(x)
    sum_xy = (x*y).sum()
    sum_xxx = (x**3).sum()
    sum_xxy = ((x**2)*y).sum()
    sum_xxxx = (x**4).sum()

    A = np.array([
        [sum_xx,  sum_x,   n],
        [sum_xxx, sum_xx,  sum_x],
        [sum_xxxx,sum_xxx, sum_xx]
    ])

    B = np.array([sum_y, sum_xy, sum_xxy])

    a, b, c = np.linalg.solve(A, B)

    y_pred = []

    for i in x:
        y_pred.append(a*i**2 + b*i + c)

    if plot:
        x_plot = np.linspace(np.sort(x)[0],np.sort(x)[-1],5000, endpoint=True)
        y_plot = [a*i**2 + b*i + c for i in x_plot]
        sns.scatterplot(x=x,y=y,label='True')
        sns.lineplot(x=x_plot,y=y_plot, color='red',label='Best Fit')
        plt.legend()
        plt.grid()
        plt.show()

    return [y_pred, a, b, c]

#exponential fit

def exponential_fit(x,y,plot=False):
    if len(x) != len(y):
        raise ValueError("Dependent and Independent variables dimension mismatch")
    
    if isinstance(x, np.ndarray):
        pass
    else:
        x = np.array(x)

    if isinstance(y, np.ndarray):
        pass
    else:
        y = np.array(y)

    y_log = np.log10(y)

    linear_result = linear_fit(x, y_log) 
    B = linear_result[1]  
    A = linear_result[2] 

    a = 10 ** A
    b = B / math.log10(math.e) 
    y_pred = a * np.exp(b * x)

    if plot:
        x_plot = np.linspace(np.sort(x)[0],np.sort(x)[-1],5000, endpoint=True)
        y_plot = [a*math.exp(b*i) for i in x_plot]
        sns.scatterplot(x=x,y=y,label='True')
        sns.lineplot(x=x_plot,y=y_plot, color='red',label='Best Fit')
        plt.legend()
        plt.grid()
        plt.show()

    return [y_pred.tolist(), a, b]

#polynomial fit

def polynomial_fit(x,y,plot=False):
    if len(x) != len(y):
        raise ValueError("Dependent and Independent variables dimension mismatch")
    
    if isinstance(x, np.ndarray):
        pass
    else:
        x = np.array(x)

    if isinstance(y, np.ndarray):
        pass
    else:
        y = np.array(y)

    y_log = np.log10(y)

    linear_result = linear_fit(x, y_log) 
    B = linear_result[1]  
    A = linear_result[2] 

    a = 10 ** A
    b = 10 ** B
    y_pred = a * b**x 

    if plot:
        x_plot = np.linspace(np.sort(x)[0],np.sort(x)[-1],5000, endpoint=True)
        y_plot = [a*b**i for i in x_plot]
        sns.scatterplot(x=x,y=y,label='True')
        sns.lineplot(x=x_plot,y=y_plot, color='red',label='Best Fit')
        plt.legend()
        plt.grid()
        plt.show()

    return [y_pred.tolist(), a, b]