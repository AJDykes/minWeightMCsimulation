# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:00:57 2025

@author: adyke

Apple Mass SKU simulation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

def load_apple_data(filename):
    """Loads apple data from given csv file"""
    apples_df = pd.read_csv(filename)
    return apples_df

def classify(apples_df, diameter_bins, elongation_bins):
    """classifies apples into bins"""
    def classify_row(row):
        diameter = row['EQ']
        elongation = row['EQ'] / row['SL']
        
        diameter_bin = next((b[2] for b in diameter_bins if b[0] <= diameter < b[1]), 'unknown')
        elongation_bin = next((b[2] for b in elongation_bins if b[0] <= elongation < b[1]), 'unknown')
        
        return pd.Series([diameter_bin, elongation_bin])
    
    apples_df[['diameter_bin', 'elongation_bin']] = apples_df.apply(classify_row, axis=1)
    apples_df['SKU'] = apples_df['diameter_bin']
    return apples_df

def monte_carlo_simulation(apples_df, sku, n_trials, pack_size):
    """monte carlo simulation for SKU packing"""
    results = []
    for _ in range(n_trials):
        subset = apples_df[apples_df['SKU'] == sku]
        sample = subset.sample(n=min(len(subset), pack_size), replace=False)
        total_mass = sample['WEIGHT'].sum()
        results.append(total_mass)
    return results 

def monte_carlo_analysis(simulation_results):
    """determines min net tube mass"""
    plt.hist(simulation_results, bins=50, density=True, alpha=0.6, color='g')
    plt.title("Monte Carlo Simulation Results")
    plt.xlabel("Total Pack Mass (m)")
    plt.ylabel("Frequency")
    plt.show()
    
    mean, std = norm.fit(simulation_results)
    if abs(std) < 0.1 * mean:
        p_0001 = norm.ppf(0.0001, loc=mean, scale=std)
        print(f"Minimum Net Tube Mass (P<0.0001): {p_0001:.2f} g")
    else:
        print("Normal approx does not hold")
        
def plot_density(apples_df):
    """plots density histogram"""
    apples_df['density'] = apples_df['WEIGHT'] / apples_df['VOLUME']
    filtered_df = apples_df[(apples_df['density'] >= 0.7) & (apples_df['density'] <= 1.1)]
    
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_df['density'], bins=50, color='g', alpha=0.7, edgecolor='black')
    plt.title('Density')
    plt.xlabel("Density (g/cm^3)")
    plt.ylabel("Frequency")
    plt.show()
   
def plot_elongation_vs_mass(apples_df):
    """plots 2d histogram of elongation vs mass"""
    plt.figure(figsize=(10,6))
    filtered_dfV = apples_df[(apples_df['EQ'] )]
    plt.scatter(apples_df['VOLUME'], apples_df['WEIGHT'])
    plt.title('Equatorial diameter vs Mass')
    plt.xlabel('Equatorial Diameter (mm)')
    plt.ylabel('Mass (Weight)')
    plt.show()
   
diameter_bins = [
    (0.0, 44.0, 'undersize'),
    (44.0, 47.0, 'small samples'),
    (47.0, 51.2, '53/5'),
    (51.2, 54.4, '58/5'),
    (54.4, 56.1, '63small'),
    (56.1, 59.0, '63medium'),
    (59.0, 61.7, '63big'),
    (61.7, 66.2, '67/4'),
    (66.2, 71.8, '72/4'),
    (71.8, 79.0, 'large sample'),
    (79.0, 100, 'oversize')
]

elongation_bins = [
    (0.000, 0.865, 'squat'),
    (0.865, 0.942, 'normal'),
    (0.942, 0.960, 'long'),
    (0.960, 1.200, 'penguin')
]



filename = "massCorrelation.csv"
apples_df = load_apple_data(filename)

apples_df = classify(apples_df, diameter_bins, elongation_bins)

#plot_density(apples_df)

#plot_elongation_vs_mass(apples_df)

#sku = '67/4'
#n_trials = 1000
#pack_size = 4
#simulation_results = monte_carlo_simulation(apples_df, sku, n_trials, pack_size)
#monte_carlo_analysis(simulation_results)

