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


dataset = 'UJI1'
clean_db = 'cleaned_db_main/CSV/'
org_db ='dataset/'

df_clean = pd.read_csv(os.path.join(clean_db, dataset, 'Train.csv'))
df_original = pd.read_csv(os.path.join(org_db, dataset, 'Train.csv'))
df_test = pd.read_csv(os.path.join(org_db, dataset, 'Test.csv'))

y_train_c = df_clean#.iloc[:,-5:]
y_train_o = df_original#.iloc[:,-5:]

df_diff = pd.concat([y_train_c,y_train_o]).drop_duplicates(keep=False)

df_fp = df_diff.iloc[:,:-5]
df_fp_o = y_train_o.iloc[:,:-5]

num_max_rss = np.round(np.max(df_fp[df_fp != 100].count(axis=1)))
num_avg_rss = np.round(np.average(df_fp[df_fp != 100].count(axis=1)))

num_max_rss_o = np.round(np.max(df_fp_o[df_fp_o != 100].count(axis=1)))
num_avg_rss_o = np.round(np.average(df_fp_o[df_fp_o != 100].count(axis=1)))

# print(num_max_rss)
# print(num_avg_rss)
# print(num_max_rss_o)
# print(num_avg_rss_o)
# print(np.shape(y_train_o)[0] - np.shape(y_train_c)[0])
# print(Counter(df_fp[df_fp != 100].count(axis=1)))
# print(Counter(df_fp_o[df_fp_o != 100].count(axis=1)))

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

ax.set_xlabel("Longitude")
ax.set_zlabel("Altitude")
ax.set_ylabel("Latitude")

ax.scatter(y_train_c["LONGITUDE"],  y_train_c["LATITUDE"], y_train_c["ALTITUDE"], color='#943126', marker='X')
ax.scatter(df_diff["LONGITUDE"],  df_diff["LATITUDE"], df_diff["ALTITUDE"], color='#2ECC71', marker='X')
# ax.scatter(df_test["LONGITUDE"],  df_test["LATITUDE"], df_test["ALTITUDE"], color='#2980B9')

ax.set_title('Removed fingerprints from the original dataset - ' + dataset)

main_path = os.path.join('report', 'PLOTS')

if not os.path.exists(main_path):
    os.makedirs(main_path)

#plt.subplots_adjust(top=0.5)
plt.savefig(os.path.join(main_path, 'fig_'+dataset+'_removed_fps.pdf'))
plt.show()