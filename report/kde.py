import os
from matplotlib import axis
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")
sns.color_palette("husl", 8)


dataset = 'TUT1'
clean_db_path = 'cleaned_db_main/CSV/'
org_db_path ='dataset/'

df_clean = pd.read_csv(os.path.join(clean_db_path, dataset, 'Train.csv'))
df_original = pd.read_csv(os.path.join(org_db_path, dataset, 'Train.csv'))

# KDE
# Flatten only the RSS values
X_train_c = df_clean.iloc[:,:-5].values
flatten_data_c = X_train_c.flatten()
X_train_o = df_original.iloc[:,:-5].values
flatten_data_o = X_train_o.flatten()

density_plot1 = sns.kdeplot(flatten_data_o , bw_method = 0.5 , fill = True,  label="Original", color='#d84797')
density_plot2 = sns.kdeplot(flatten_data_c , bw_method = 0.5 , fill = True,  label="Reduced", color='#3ABEFF')

# density_plot1.set_title('('+str(num_plot)+')')
density_plot1.set_title('(a)')
density_plot1.set_xlabel("RSS values")


plt.title(dataset + " - Kernel Density Estimation (KDE)", fontsize=10)
plt.subplots_adjust(hspace=0.6)
plt.subplots_adjust(bottom=0.25)
plt.legend(loc="upper left")
# plt.legend(ncol=1, loc="upper left", frameon=True)

main_path = os.path.join('report', 'PLOTS')

if not os.path.exists(main_path):
    os.makedirs(main_path)

plt.savefig(os.path.join(main_path,'fig_'+'-'.join(dataset)+'-KDE.pdf'))
plt.show()