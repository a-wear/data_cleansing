import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")
sns.color_palette("husl", 8)

file = 'report/results.csv'
df_results  =  pd.read_csv(file)
#df_results['TIME_ORG'] = df_results['TIME_ORG'] *100
fig, ax = plt.subplots()
ax1 = sns.regplot(x='DIM_ORG', y='TIME_ORG', data=df_results, ci=None,  marker="x", label='Original Dataset')
ax2 = sns.regplot(x='DIM_RED', y='TIME_RED', data=df_results, ci=None, marker="+", label='Cleaned Dataset')
# ax.set_aspect('auto')
ax.set_title('Data Dimension vs. Testing Time')
ax.set_xlabel('Data Size')
ax.set_ylabel('Time [s]')
# ax1.set(yscale="log")
# ax2.set(yscale="log")
plt.legend()
fig.tight_layout()
plt.show()