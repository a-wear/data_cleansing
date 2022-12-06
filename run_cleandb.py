import pandas as pd

from preprocessing.data_preprocessing import load, new_non_detected_value
from preprocessing.data_representation import DataRepresentation
from miscellaneous.misc import Misc
from sklearn.preprocessing import LabelEncoder
from preprocessing.data_cleaning import clean_dataset
from positioning.position import Position_KNN
import time as ti
from datetime import datetime
import joblib
import logging
import os
import numpy as np


def run_cleandb(dataset_name=None, path_config=None, dataset_config=None, model_config=None):
    """_summary_

    Args:
        dataset_name (string, required): Dataset name, e.g., TUT1, UJI1, etc. Defaults to None.
        path_config (dic, required): Datasets, cleaned datasets, results paths. Defaults to None.
        dataset_config (dic, required): Dataset configuration parameters. Defaults to None.
        model_config (dic, required): Model configuration. Defaults to None.
    """

    misc = Misc()
    
    print(misc.log_msg('WARNING', 'Cleaning dataset...'))
    
    # Setting paths
    dataset_path = os.path.join(path_config['data_source'], dataset_name)
    main_path_save = os.path.join(path_config['saved_model'], dataset_config['name'], 'CLEANED_DB')

    # Loading datasets
    if bool(dataset_config['train_dataset']):
        X_train, y_train = load(os.path.join(dataset_path, dataset_config['train_dataset']))

    if bool(dataset_config['test_dataset']):
        X_test, y_test = load(os.path.join(dataset_path, dataset_config['test_dataset']))

    if bool(dataset_config['validation_dataset']):
        X_valid, y_valid = load(os.path.join(dataset_path, dataset_config['validation_dataset']))
    else:
        X_valid = []
        y_valid = []

    X_train_org = X_train
    X_test_org = X_test
    X_valid_org = X_valid
    
    # Change data representation
    new_non_det_val = new_non_detected_value(X_train, X_test, X_valid)
    dr = DataRepresentation(x_train=X_train, x_test=X_test, x_valid=X_valid,
                            type_rep=dataset_config['data_representation'],
                            def_no_val=dataset_config['default_null_value'],
                            new_no_val=new_non_det_val)
    X_train, X_test, X_valid = dr.data_rep()
    
    # new_non_det_val_2 = new_non_detected_value(X_train, X_test, X_valid)

    y_train_org = y_train.copy()
    
    # Floor label encoding
    encoding = LabelEncoder()
    y_train['FLOOR'] = encoding.fit_transform(y_train.iloc[:, 3])
    y_test['FLOOR'] = encoding.transform(y_test.iloc[:, 3])

    # Position estimation - Baseline
    position = Position_KNN(k=dataset_config['k'], metric=dataset_config['distance_metric'])
    position.fit(X_train, y_train.values)
    t1 = ti.time()
    floor_hit_rate_org, true_false_values_org, pred_fhr_org = position.floor_hit_rate(X_test, y_test.values)
    building_hit_rate_org, pred_bhr_org = position.building_hit_rate(X_test, y_test.values)
    error2D_org, error2D_values_org = position.predict_position_2D(X_test, y_test.values, true_floors=true_false_values_org)
    error3D_org, error3D_values_org = position.predict_position_3D(X_test, y_test.values)
    prediction_time_org = ti.time()-t1
    
    rep_pred_org = pd.DataFrame(np.concatenate((
                    np.array(error3D_values_org, ndmin=2).T,
                    np.array(pred_fhr_org, ndmin=2).T,
                    np.array(pred_bhr_org, ndmin=2).T), axis=1), columns=['ERROR_3D','PRED_FLOOR','PRED_BUILDING'])

    cleaned_db = os.path.join(path_config['cleaned_db'], 'CSV', dataset_config['name'], 'Train.csv')
    
    # Data cleansing
    if dataset_config['clean_db']:
        t2 = ti.time()
        clean_dataset(org_x_train=pd.DataFrame(X_train_org), org_y_train=y_train_org,
                                        preprocessed_x_train=pd.DataFrame(X_train), db_config=dataset_config,
                                        path_config=path_config, model_config=model_config)
        X_train, y_train = load(cleaned_db)
        dr = DataRepresentation(x_train=X_train, x_test=X_test_org, x_valid=X_valid_org,
                            type_rep=dataset_config['data_representation'],
                            def_no_val=dataset_config['default_null_value'],
                            new_no_val=new_non_det_val)
        X_train, X_test, X_valid = dr.data_rep()
        
        cleaning_time = ti.time() - t2
    else:
        if os.path.exists(cleaned_db):
            X_train, y_train = load(cleaned_db)
            dr = DataRepresentation(x_train=X_train, x_test=X_test_org, x_valid=X_valid_org,
                            type_rep=dataset_config['data_representation'],
                            def_no_val=dataset_config['default_null_value'],
                            new_no_val=new_non_det_val)
            X_train, X_test, X_valid = dr.data_rep()
        else:
            print(misc.log_msg('ERROR', 'Error: Please change the "clean_db" parameter to ** TRUE ** in the config file to generate the cleaned DB.'))
            exit(-1)
    
    y_train['FLOOR'] = encoding.transform(y_train.iloc[:, 3])
    
    # Position estimation - data cleaned
    position.fit(X_train, y_train.values)
    t3 = ti.time()
    floor_hit_rate, true_false_values, pred_fhr = position.floor_hit_rate(X_test, y_test.values)
    building_hit_rate, pred_bhr = position.building_hit_rate(X_test, y_test.values)
    error2D, error2D_values = position.predict_position_2D(X_test, y_test.values, true_floors=true_false_values)
    error3D, error3D_values = position.predict_position_3D(X_test, y_test.values)
    prediction_time_cleaned = ti.time() - t3

    rep_pred = pd.DataFrame(np.concatenate((
                    np.array(error3D_values, ndmin=2).T,
                    np.array(pred_fhr, ndmin=2).T,
                    np.array(pred_bhr, ndmin=2).T), axis=1), columns=['ERROR_3D','PRED_FLOOR','PRED_BUILDING'])

    # Saving results
    datestr = "%m/%d/%Y %I:%M:%S %p"
    save_path_log = os.path.join("results", dataset_config['name'], "LOG")
    save_path_results = os.path.join("results", dataset_config['name'], "RESULTS")

    if not os.path.exists(save_path_log):
        os.makedirs(save_path_log)
    
    if not os.path.exists(save_path_results):
        os.makedirs(save_path_results)
        
    rep_pred.to_csv(os.path.join(save_path_results,'PRED_CLEAN.csv'),index=False)
    rep_pred_org.to_csv(os.path.join(save_path_results,'PRED_FULLDB.csv'),index=False)

    logging.basicConfig(
        filename=save_path_log + '/' + datetime.today().strftime('%Y-%m-%d-%H:%M:%S') + '.log',
        level=logging.INFO,
        filemode="w",
        datefmt=datestr,
        # force=True
    )


    logging.info("---------------------------- KNN ---------------------------")
    logging.info(' Dataset : ' + dataset_name)
    logging.info(' Original database size : {}'.format(np.shape(X_train_org)))
    logging.info(' Mean 2D positioning error : {:.3f}m'.format(error2D_org))
    logging.info(' Mean 3D positioning error : {:.3f}m'.format(error3D_org))
    logging.info(' Floor hit rate : {:.3f}%'.format(floor_hit_rate_org))
    logging.info(' Building hit rate : {:.3f}%'.format(building_hit_rate_org))
    logging.info(' Full time : {:.6f}s'.format(prediction_time_org))
    logging.info(' --------------------------------------------------------- ')
    logging.info(' After the data cleaning : {}'.format(np.shape(X_train)))
    logging.info(' Configured match percent: {}'.format(dataset_config['match_percentage']))
    logging.info(' Reduced : {:.2f}%'.format(100-(np.shape(X_train)[0]*100/np.shape(X_train_org)[0])))
    if dataset_config['clean_db'] == 'True':
        logging.info(' Cleaning time : {:.6f}s'.format(cleaning_time))
    logging.info(' Mean 2D positioning error : {:.3f}m'.format(error2D))
    logging.info(' Mean 3D positioning error : {:.3f}m'.format(error3D))
    logging.info(' Floor hit rate : {:.3f}%'.format(floor_hit_rate))
    logging.info(' Building hit rate : {:.3f}%'.format(building_hit_rate))
    logging.info(' Full time : {:.6f}s'.format(prediction_time_cleaned))
    logging.info(' ------- KNN configuration ------- ')
    logging.info(" model: kNN")
    logging.info(" K: " + str(dataset_config['k']))
    logging.info(" distance: " + dataset_config['distance_metric'])

    print(" Dataset: " + dataset_config['name'])
    print(' Original database size (Training DB) : {}'.format(np.shape(X_train_org)))
    print(' --------------------------------------------------------- ')
    print(' Mean 2D positioning error : {:.3f}m'.format(error2D_org))
    print(' Mean 3D positioning error : {:.3f}m'.format(error3D_org))
    print(' Floor hit rate : {:.3f}%'.format(floor_hit_rate_org))
    print(' Building hit rate : {:.3f}%'.format(building_hit_rate_org))
    print(' Full time : {:.6f}s'.format(prediction_time_org))
    print(' --------------------------------------------------------- ')
    print(' After the data cleaning (Training DB) : {}'.format(np.shape(X_train)))
    print(' Configured match percent: {}%'.format(dataset_config['match_percentage']))
    print(' Reduced : {:.2f}%'.format(100 - (np.shape(X_train)[0] * 100 / np.shape(X_train_org)[0])))
    if dataset_config['clean_db'] == 'True':
        print(' Cleaning time : {:.6f}s'.format(cleaning_time))
    print(" Mean 2D positioning error: {:.3f}m".format(error2D))
    print(" Mean 3D positioning error: {:.3f}m".format(error3D))
    print(" Floor hit rate: {:.3f}%".format(floor_hit_rate))
    print(" Building hit rate: {:.3f}%".format(building_hit_rate))
    print(" Full time : {:.6f}s".format(prediction_time_cleaned))
