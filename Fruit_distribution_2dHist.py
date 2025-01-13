# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:55:08 2025

@author: adyke

Mean Size Summary 2d histogram
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_fruit_distribution(input_csv, output_csv, num_simulations=1000000):
    """
    Simulate fruit based on bivariate normal distribution and segment the results.

    Parameters:
        input_csv (str): Path to the input CSV file containing distribution parameters.
        output_csv (str): Path to save the segmented simulation results.
        num_simulations (int): Number of fruit pieces to simulate.
    """
    try:
        params = pd.read_csv(input_csv)

        mean_eq = params['meanEQ'][0]
        sd_eq = params['sdEQ'][0]
        mean_elong = params['meanElong'][0]
        sd_elong = params['sdElong'][0]
        covariance = params['covar'][0]
        
        cov_matrix = [[sd_eq**2, covariance], [covariance, sd_elong**2]]

        mean_vector = [mean_eq, mean_elong]

        simulated_data = np.random.multivariate_normal(mean_vector, cov_matrix, num_simulations)
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

        segmented_data = pd.DataFrame({
            'EquatorialDiameter': equatorial_diameter,
            'Elongation': elongation,
            'DiameterCategory': [classify(d, diameter_bins) for d in equatorial_diameter],
            'ElongationCategory': [classify(e, elongation_bins) for e in elongation]
        })

        segmented_data.to_csv(output_csv, index=False)
        print(f"Done, results saved as {output_csv}")
        
        plt.figure(figsize=(10, 8))
        hist, xedges, yedges, im = plt.hist2d(equatorial_diameter, elongation, bins=100, cmap=plt.cm.jet)
        
        plt.colorbar(im, label='Frequency')
        plt.xlabel('Equatorial Diameter')
        plt.ylabel('Elongation')
        plt.title('2D Histogram of EQ vs Elongation')
        plt.show()
        
    except Exception as e:
        print(f"Not good enough buddy!: {e}")
        
        
simulate_fruit_distribution('MeanSizeSummaries.csv', 'output_simulation2.csv')        
