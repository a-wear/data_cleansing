# libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file = 'report/results.csv'
df_results  =  pd.read_csv(file)
 
# set width of bars
barWidth = 0.25
 
# set heights of bars
bars1 = [12, 30, 1, 8, 22]
bars2 = [28, 6, 16, 5, 10]
bars3 = [29, 3, 24, 25, 17]
 
# Set position of bar on X axis
r1 = np.arange(len(df_results['TIME_ORG']))
r2 = [x + barWidth for x in r1]

 
# Make the plot
plt.bar(r1, df_results['TIME_ORG'], color='#7f6d5f', width=barWidth, edgecolor='white', label='Original data')
plt.bar(r2, df_results['TIME_RED'], color='#557f2d', width=barWidth, edgecolor='white', label='Cleaned Data')

 
# Add xticks on the middle of the group bars
plt.xlabel('Dataset', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(df_results['TIME_ORG']))], df_results['DATASET'])
plt.yscale("log")
# Create legend & Show graphic
plt.legend()
plt.show()