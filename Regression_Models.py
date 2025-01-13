# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:52:22 2025

@author: adyke
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def linear_regression(filename):
    """calculates a linear regression for eqatorial distance vs mass"""
    data = pd.read_csv(filename)
    
    x = data['EQ'].values.reshape(-1, 1)
    y = data['WEIGHT'].values
    
    regr = LinearRegression().fit(x, y)
    r_sq = regr.score(x, y)
    
    y_pred = regr.predict(x)
    
    plt.scatter(x, y, color='blue', alpha=0.1, label='Data Points')
    plt.plot(x, y_pred, color='red', label='Linear regression line')
    plt.xlabel('Equatorial Distance')
    plt.ylabel('Weight')
    plt.title('Linear Regression of Equatorial Distance vs Weight')
    plt.show()
    print(f"R^2: {r_sq} \n intercept: {regr.intercept_} \n slope: {regr.coef_}")
 
def load_and_clean_data(filename):
    """loads and cleans values from csv file to process in regression"""
    data = pd.read_csv(filename)
    data = data.dropna(subset=['EQ', 'WEIGHT', 'VOLUME'])
    
    data['DENSITY'] = data['WEIGHT'] / data['VOLUME']
    
    data = data[(data['DENSITY'] >= 0.7) & (data['DENSITY'] <= 1.1)]
    sorted_data = data.sort_values(by='EQ')
    return sorted_data
 

def polynomial_regression(filename, degree=3):
    """Plots a polynomial regression for equatorial distance vs mass"""
    data = load_and_clean_data(filename)
    
    X = data['EQ'].values.reshape(-1,1)
    y = data['WEIGHT'].values

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    regr = LinearRegression().fit(X_poly, y)
    y_pred = regr.predict(X_poly)
    
    plt.scatter(X, y, color='blue', alpha=0.05, s=10, label='Data points')
    plt.plot(X, y_pred, color='red', label=f'polynomial regression (degree {degree})')
    plt.xlabel('Equatorial Distance (mm)')
    plt.ylabel('Weight (g)')
    plt.title(f'Polynomial Regression of EQ vs Weight (degree {degree})')
    plt.legend()
    plt.show()    
    print(f"Polynomial Coefficients: {regr.coef_}")
    print(f"Intercept: {regr.intercept_}")




filename = "massCorrelation.csv"

#linear_regression(filename)

polynomial_regression(filename, degree=3)
