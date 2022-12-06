from cProfile import label
from collections import Counter
import os
from turtle import color
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")
sns.color_palette("husl", 8)


dataset = 'UJI2'
clean_db = pd.read_csv('results/'+ dataset +'/RESULTS/PRED_CLEAN.csv')
full_db = pd.read_csv('results/'+ dataset +'/RESULTS/PRED_FULLDB.csv')

plt.rcParams["figure.figsize"] = [5.5, 3]
plt.rcParams["figure.autolayout"] = True

count, bins_count = np.histogram(clean_db['ERROR_3D'], bins=30)
pdf = count / sum(count)
cdf = np.cumsum(pdf)

count_fdb, bins_count_fdb = np.histogram(full_db['ERROR_3D'], bins=30)
pdf_fdb = count_fdb / sum(count_fdb)
cdf_fdb = np.cumsum(pdf_fdb)

plt.plot(bins_count_fdb[1:], cdf_fdb, label="Original", color='#d84797')
plt.plot(bins_count[1:], cdf, label="Reduced", color='#3ABEFF')
plt.title(dataset + " - CDF 3D positioning error")
plt.xlabel("3D positioning error [m]")
plt.ylabel("CDF")
plt.legend()

main_path = os.path.join('report', 'PLOTS')

if not os.path.exists(main_path):
    os.makedirs(main_path)
    
plt.savefig(os.path.join(main_path,'fig_'+dataset+'_cdf.pdf'))
plt.show()