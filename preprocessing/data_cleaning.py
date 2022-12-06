import warnings
from collections import Counter
warnings.filterwarnings("ignore")
import os
import pandas as pd
import numpy as np
from miscellaneous.misc import Misc
from scipy.io import savemat

# Matching function
def match_function(row, position, dataframe, num_max_rss, m_percent):
    intersect = dataframe.apply(lambda x: list(set(x).intersection(set(row))),axis=1)
    intersect[position] = []
    intersect = list(map(lambda x: list(filter(lambda y: y > 0,x)), intersect))
    match_percent = list(map(lambda x: 0 if (len(x)*100)/num_max_rss < m_percent else (len(x)*100)/num_max_rss, intersect))
    return match_percent

# Sorting data - descending
def sort_df(df):
    return pd.DataFrame(
        data=df.columns.values[np.argsort(-df.values, axis=1)],
        columns=[i for i in range(df.shape[1])]
    )

# Sorting data - ascending
def sort_asc_df(df):
    return pd.DataFrame(
        data=df.columns.values[np.argsort(df.values, axis=1)],
        columns=[i for i in range(df.shape[1])]
    )

def clean_dataset(org_x_train=None, org_y_train=None, preprocessed_x_train=None, db_config=None,
                  path_config=None, model_config=None):
    """_summary_

    Args:
        org_x_train (dataframe, optional): Original train set. Defaults to None.
        org_y_train (dataframe, optional): Original labels train set. Defaults to None.
        preprocessed_x_train (dataframe, optional): Preprocessed dataset. Defaults to None.
        db_config (dic, optional): Dataset configuration. Defaults to None.
        path_config (dic, optional): General paths. Defaults to None.
        model_config (dic, optional): Model configuration. Defaults to None.
    """
    misc = Misc()
    if org_x_train is None or preprocessed_x_train is None:
        misc.log_msg('ERROR', 'Error: Original dataset and/or preprocessed dataset have not been defined.')
        exit(-1)

    match_percentage = db_config['match_percentage']

    # Determine the average number of RSS values in the dataset
    df_x_train = preprocessed_x_train.copy()
    if model_config['wp'] == "MAX":
        num_max_rss = np.max(df_x_train[df_x_train > 0].count(axis=1))
    elif model_config['wp'] == "AVG":
        num_max_rss = np.round(np.average(df_x_train[df_x_train > 0].count(axis=1)))
    else:
        print(misc.log_msg("ERROR", "Sorry, method " + model_config['wp'] + " not allowed."))
        exit(-1)

    # Sorting the dataset (Descending)
    sorted_df = np.sort(df_x_train)[:, ::-1]
    sorted_df = pd.DataFrame(np.where(sorted_df > 0, 1, -1))
    temp = sort_df(df_x_train).replace(0, df_x_train.shape[1] + 1)
    temp2 = temp.mul(sorted_df).values
    temp2 = pd.DataFrame(np.where(temp2 < 0, 0, temp2)).iloc[:, 0:int(num_max_rss - 1)]

    # Match percentage
    df_match = pd.DataFrame()
    for i in range(0, temp2.shape[0]):
        df_match[i] = match_function(temp2.iloc[i, :], i, temp2, num_max_rss, match_percentage)

    # Getting fingerprints with zero match percentage
    df_idx = pd.DataFrame(df_match.max(axis=1))
    index_zeros = df_idx.index[df_idx[0] == 0].tolist()

    X_train_proc = df_x_train.drop(index_zeros).values

    # Removing unnecesary fingerprints
    X_train = pd.DataFrame(org_x_train)
    X_train = X_train.drop(index_zeros)
    y_train = org_y_train.drop(index_zeros)

    # Saving cleansed dataset (.mat and .csv)
    database = {}
    test_db = pd.read_csv(
        os.path.join(path_config['data_source'], db_config['name'], db_config['test_dataset']))

    path_save_mat = os.path.join(path_config['cleaned_db'], 'MAT', db_config['name'])
    path_save_csv = os.path.join(path_config['cleaned_db'], 'CSV', db_config['name'])

    if not os.path.exists(path_save_mat):
        os.makedirs(path_save_mat)

    if not os.path.exists(path_save_csv):
        os.makedirs(path_save_csv)

    database['trainingMacs'] = X_train.values
    database['trainingLabels'] = y_train.values

    f_train_dataset = pd.concat([pd.DataFrame(X_train), y_train], axis=1)

    f_train_dataset.to_csv(os.path.join(path_save_csv, 'Train.csv'), index=False)
    test_db.to_csv(os.path.join(path_save_csv, 'Test.csv'), index=False)

    database['testMacs'] = test_db.iloc[:, :-5].values
    database['testLabels'] = test_db.iloc[:, -5:].values

    if db_config['validation_dataset'] != '':
        validation_db = pd.read_csv(
            os.path.join(path_config['data_source'], db_config['name'], db_config['validation_dataset']))
        database['validationMacs'] = validation_db.iloc[:, :-5].values
        database['validationLabels'] = validation_db.iloc[:, -5:].values
        validation_db.to_csv(os.path.join(path_save_csv, 'Validation.csv'), index=False)

    db = {}
    db['database'] = database
    path_mat = os.path.join(path_save_mat, db_config['name'] + '.mat')
    savemat(path_mat, db)

