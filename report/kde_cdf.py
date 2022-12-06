import os
from matplotlib import axis
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")
sns.color_palette("husl", 8)


datasets = ['LIB2', 'UJI1'] # Minimum 2 datasets 
clean_db_path = 'cleaned_db/CSV/'
org_db_path ='dataset/'

fig, axes = plt.subplots(2, len(datasets))
num_plot = 1
for i in range(0, len(datasets)):
    # Load the datasets
    df_clean = pd.read_csv(os.path.join(clean_db_path, datasets[i], 'Train.csv'))
    df_original = pd.read_csv(os.path.join(org_db_path, datasets[i], 'Train.csv'))

    # KDE
    # Flatten only the RSS values
    X_train_c = df_clean.iloc[:,:-5].values
    flatten_data_c = X_train_c.flatten()
    X_train_o = df_original.iloc[:,:-5].values
    flatten_data_o = X_train_o.flatten()

    density_plot1 = sns.kdeplot(flatten_data_o , bw_method = 0.5 , fill = True,  label="Original", color='#d84797', ax=axes[0,i])
    density_plot2 = sns.kdeplot(flatten_data_c , bw_method = 0.5 , fill = True,  label="Reduced", color='#3ABEFF', ax=axes[0,i])

    # density_plot1.set_title('('+str(num_plot)+')')
    density_plot1.set_title('(a)')
    density_plot1.set_xlabel("RSS values")
    
    if num_plot > 1:
        density_plot1.set_ylabel("")
        
    # CDF
    clean_db = pd.read_csv('results/'+ datasets[i] +'/RESULTS/PRED_CLEAN.csv')
    full_db = pd.read_csv('results/'+ datasets[i] +'/RESULTS/PRED_FULLDB.csv')

    count, bins_count = np.histogram(clean_db['ERROR_3D'], bins=30)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)

    count_fdb, bins_count_fdb = np.histogram(full_db['ERROR_3D'], bins=30)
    pdf_fdb = count_fdb / sum(count_fdb)
    cdf_fdb = np.cumsum(pdf_fdb)

    axes[1,i].plot(bins_count_fdb[1:], cdf_fdb, label="Original", color='#d84797')
    axes[1,i].plot(bins_count[1:], cdf, label="Reduced", color='#3ABEFF')
    
    # axes[1,i].set_title('('+str(num_plot+len(datasets))+')')
    axes[1,i].set_xlabel("3D positioning error [m]")

    if num_plot > 1:
        axes[1,i].set_ylabel("")
    else:
        axes[1,i].set_ylabel("CDF")
    
    num_plot += 1


density_plot2.set_title('(b)')
axes[1,0].set_title('(c)')
axes[1,1].set_title('(d)')
# Create the legend
axes[1,0].legend([density_plot1, density_plot2],     # The line objects
           labels=['Original DB', 'Cleaned DB'],   # The labels for each line
           loc= "lower center",   # Position of legend
            bbox_to_anchor=(1.1 * len(datasets)*0.5, -0.75), 
            fancybox=True, 
            shadow=False, ncol=2  # Title for the legend
           )

axes[1,i].set_xlim([0, 60])

fig.suptitle("Data Distribution (KDE) and 3D Positioning error (CDF)", fontsize=10)
plt.subplots_adjust(hspace=0.6)
plt.subplots_adjust(bottom=0.25)
# plt.legend(ncol=1, loc="upper left", frameon=True)

main_path = os.path.join('report', 'PLOTS')

if not os.path.exists(main_path):
    os.makedirs(main_path)

plt.savefig(os.path.join(main_path,'fig_'+'-'.join(datasets)+'_density_cdf.pdf'))
plt.show()