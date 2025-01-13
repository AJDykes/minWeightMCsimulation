# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:37:30 2025

@author: adyke
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns


def train_mass_model(filename, degree=3):
    """Trains model to predict mass based on EQ and elongaiton"""
    data = pd.read_csv(filename)
    data = data.dropna(subset=['EQ', 'SL', 'WEIGHT'])
    X = data[['EQ', 'SL']].values
    y = data['WEIGHT'].values
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression().fit(X_poly, y)
    
    r_squared = model.score(X_poly, y)
    return model, poly, r_squared

def simulate_fruit_distribution(input_csv, output_csv):
    """simulates fruit distribution based on input parameters"""
    params = pd.read_csv(input_csv)

    mean_eq = params['meanEQ'][0]
    sd_eq = params['sdEQ'][0]
    mean_elong = params['meanElong'][0]
    sd_elong = params['sdElong'][0]
    covariance = params['covar'][0]

    cov_matrix = [[sd_eq**2, covariance], [covariance, sd_elong**2]]
    mean_vector = [mean_eq, mean_elong]

    simulated_data = np.random.multivariate_normal(mean_vector, cov_matrix, 10000)
    equatorial_diameter = simulated_data[:, 0]
    elongation = simulated_data[:, 1]
    
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
    
    def classify(value, bins):
        for min_val, max_val, label in bins:
            if min_val <= value <= max_val:
                return label
        return 'unclassified'
    
    simulated_df = pd.DataFrame({
        'EquatorialDiameter': equatorial_diameter,
        'Elongation': elongation,
        'DiameterCategory': [classify(d, diameter_bins) for d in equatorial_diameter],
        'ElongationCategory': [classify(e, elongation_bins) for e in elongation]
    })
    
    simulated_df.to_csv(output_csv, index=False)
    return output_csv

def Mass_from_simulated(input_csv, output_csv, mass_output_csv, regression_data, degree):
    """
    Takes simulated apple data and calculates
    mass using the polynomial regression model
    """
    try:    
        model, poly, r_squared = train_mass_model(regression_data, degree)
        
        simulation_file = simulate_fruit_distribution(input_csv, output_csv)
            
        simulated_data = pd.read_csv(simulation_file)
        X_simulated = simulated_data[['EquatorialDiameter', 'Elongation']].values
        X_poly_simulated = poly.transform(X_simulated)
        
        predicted_mass = model.predict(X_poly_simulated)
        simulated_data['Mass'] = predicted_mass
        
        simulated_data.to_csv(mass_output_csv, index=False)
        print(f"Mass predictions: {mass_output_csv}")
        print(f"R Squared: {r_squared}")
        return mass_output_csv
        
    
    except Exception as e:
        print(f'U fucked up: {e}')
        return None
    
def filtered_bins(input_csv, diameter_categories, include_elongation_for_diameter, excluded_elongations, output_csv):
    """filters output csv to specified bins"""    
    data = pd.read_csv(input_csv)
    
    filtered_data = data[data['DiameterCategory'].isin(diameter_categories)]
    
    include_elongation_data = filtered_data[filtered_data['DiameterCategory'] == include_elongation_for_diameter]
    
    exclude_elongation_data = filtered_data[~filtered_data['DiameterCategory'].isin([include_elongation_for_diameter])]
    exclude_elongation_data = exclude_elongation_data[~exclude_elongation_data['ElongationCategory'].isin(excluded_elongations)]
    
    final_filtered_data = pd.concat([include_elongation_data, exclude_elongation_data])
    
    final_filtered_data.to_csv(output_csv, index=False)
    
    print(f"Filtered data saved to {output_csv}")
    return output_csv
        
def filtered_bins_2(data, diameter_conditions, output_csv):
    """alternative filter for elongation bins and diameter categories"""
    
    filtered_data = pd.DataFrame()
    
    for diameter, elongations in diameter_conditions.items():
        if elongations == 'all':
            temp_data = data[data['DiameterCategory'] == diameter]
        else:
            temp_data = data[
                (data['DiameterCategory'] == diameter) &
                (data['ElongationCategory'].isin(elongations))
            ]
        filtered_data = pd.concat([filtered_data, temp_data], ignore_index=True)
    
    filtered_data.to_csv(output_csv, index=False)
    print(f"filtered data to: {output_csv}")
    return filtered_data

def monte_carlo_min_weight(filtered_data, diameter_bins_tube_size):
    """
    Perform Monte Carlo simulation to determine
    the minimum weight for apple packs
    """
    min_weights = {}
    
    for diameter_bin, tube_size in diameter_bins_tube_size.items():
        bin_data = filtered_data[filtered_data['DiameterCategory'] == diameter_bin]
        weights = bin_data['Mass'].values
        
        if len(weights) < tube_size:
            print("Skipped bin")
            continue
        
        pack_weights = []
        for _ in range(1_000_000):  
            pack = np.random.choice(weights, size=tube_size, replace=False)
            pack_weights.append(np.sum(pack))
        
        pack_weights = np.array(pack_weights)
        min_weight = np.percentile(pack_weights, 0.01) 
        min_weights[diameter_bin] = min_weight
        
        print(f"Diameter Bin: {diameter_bin}, Tube Size: {tube_size}, Min Weight (P=0.0001): {min_weight:.2f}")
        
        plt.figure(figsize=(10, 6))
        sns.histplot(pack_weights, bins=100, kde=False, color='skyblue', stat='density', label='Pack Weights')
        
        mean, std = np.mean(pack_weights), np.std(pack_weights)
        x = np.linspace(min(pack_weights), max(pack_weights), 1000)
        pdf = norm.pdf(x, mean, std)
        plt.plot(x, pdf, 'r', label=f'Normal Fit\nMean: {mean:.2f}, Std: {std:.2f}')
        
        plt.title(f'Pack Weight Distribution for {diameter_bin} (Tube Size: {tube_size})', fontsize=14)
        plt.xlabel('Pack Weight', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(fontsize=10)
        
        plt.axvline(min_weight, color='green', linestyle='--', label=f'Min Weight (P=0.0001): {min_weight:.2f}')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    return min_weights


Mass_from_simulated(
    input_csv='MeanSizeSummaries.csv',
    output_csv='segmented_intermediate.csv',
    mass_output_csv='segmented_data_w_mass8.csv',
    regression_data='massCorrelation.csv',
    degree=3
)

#EXAMPLE INPUT FOR "filtered_bins()"

#filtered_bins(
#    'segmented_data_w_mass3.csv',      # Input CSV
#    ['63medium', '63big', '63small'], # Diameter categories
#    '63big',                         # Keep long and penguin elongation for '63small'
#    ['squat', 'normal'],               # Elongations to exclude from other diameter categories
#    'filtered_diameters2.csv'          # Output CSV
#)

#filtered_data = pd.read_csv('filtered_diameters5.csv')  

filtered_data = filtered_bins_2(
    data=pd.read_csv('segmented_data_w_mass8.csv'),  # Load data
    diameter_conditions={
        '63big': 'all', # Include all elongation bins
        '63medium': ['normal', 'long', 'penguin'],  # Include 3 elongation bins
        '63small': ['long', 'penguin']  # Include 2 elongation bins
    },
    output_csv='filtered_data_23.csv'
)


diameter_bins_tube_size = {
    '63medium': 4,  # Tube size 4 for '63medium'
    '63large': 5,   # etc..
    '63small': 4    
}

min_weights = monte_carlo_min_weight(filtered_data, diameter_bins_tube_size)

print("Minimum weights for each diameter bin:")
print(min_weights)







