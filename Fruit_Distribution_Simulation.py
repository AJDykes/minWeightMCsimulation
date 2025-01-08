# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:01:26 2024

@author: adyke

Mean Size Summary Simulation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

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
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        hist, xedges, yedges = np.histogram2d(equatorial_diameter, elongation, bins=30)
        
        xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = np.zeros_like(xpos)
        
        dx = dy = 0.5 * np.ones_like(zpos)
        dz = hist.ravel()
        
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
        
        ax.set_xlabel('Equatorial Diameter')
        ax.set_ylabel('Elongation')
        ax.set_zlabel('Frequency')
        ax.set_title('3D Histogram of Equatorial Diameter vs Elongation')
        plt.show()
        
    except Exception as e:
        print(f"Not good enough buddy!: {e}")


def monte_carlo_analysis(input_csv, num_simulations=10):
    """
    Perform Monte Carlo analysis on the segmented simulation data.

    Parameters:
        input_csv (str): Path to the input CSV file containing segmented simulation data.
        num_simulations (int): Number of Monte Carlo simulations to perform.
    
    Returns monte carlo data frame
    """
    
    try:
        data = pd.read_csv(input_csv)
        results = {}

        for i in range(num_simulations):
            sample = data.sample(frac=1, replace=True)

            diameter_counts = sample['DiameterCategory'].value_counts()
            elongation_counts = sample['ElongationCategory'].value_counts()

            for category in diameter_counts.index:
                if category not in results:
                    results[category] = []
                results[category].append(diameter_counts[category])

            for category in elongation_counts.index:
                if category not in results:
                    results[category] = []
                results[category].append(elongation_counts[category])

        summary_stats = {
            category: {
                'mean': np.mean(values),
                'std_dev': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            } for category, values in results.items()
        }

        summary_df = pd.DataFrame(summary_stats).T

        return summary_df

    except Exception as e:
        print(f"Try again bitch: {e}")
        return None
    
simulate_fruit_distribution('MeanSizeSummaries.csv', 'output_simulation2.csv')
monte_carlo_results = monte_carlo_analysis('output_simulation2.csv')
print(monte_carlo_results)