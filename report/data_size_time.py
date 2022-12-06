import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")
sns.color_palette("husl", 8)

file = 'report/results.csv'
df_results  =  pd.read_csv(file)

# fig, axes = plt.subplots(2, 1)

bar_plot1 = sns.barplot(x='DATASET', y='DIM_ORG', data=df_results, label="Original Size", color='#d84797')
bar_plot2 = sns.barplot(x='DATASET', y='DIM_RED', data=df_results, label="Reduced Size", color='#3ABEFF')

#df_results['TIME_ORG'] =  df_results['TIME_ORG']*100
#df_results['TIME_RED'] =  df_results['TIME_RED']*100

#bar_plot3 = sns.lineplot(x='DATASET', y='TIME_ORG', data=df_results, label="Original Time", color='#d84797', ax=axes[1], linestyle='--')
#bar_plot4 = sns.lineplot(x='DATASET', y='TIME_RED', data=df_results, label="Reduced Time", color='#3ABEFF', ax=axes[1], linestyle=':')

bar_plot1.set_xlabel("Dataset")
bar_plot1.set_ylabel("Num. Samples")
#bar_plot3.set_ylabel("Time [s]")

bar_plot1.set(yscale="log")
bar_plot2.set(yscale="log")
#bar_plot3.set(yscale="log")
#bar_plot4.set(yscale="log")

#fig.autofmt_xdate(rotation=45 )
plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.40)
plt.legend(loc= "lower center", bbox_to_anchor=(0.5, -0.45), fancybox=True, shadow=False, ncol=2)
# plt.legend([bar_plot1, bar_plot2],     # The line objects
#            labels=['Original DB', 'Cleaned DB'],   # The labels for each line
#            loc= "lower center",   # Position of legend
#             bbox_to_anchor=(0.5, -0.3), 
#             fancybox=True, 
#             shadow=False, ncol=2  # Title for the legend
#            )
main_path = os.path.join('report', 'PLOTS')

if not os.path.exists(main_path):
    os.makedirs(main_path)
    
plt.savefig(os.path.join(main_path, 'fig_dims.pdf'))
plt.show()