## create the confusion matrix dataframe
import numpy as np
import pandas as pd
import json

K_values = [1, 2, 5, 10, 15, 20, 50, 100, 500, 1000]
prefixes = ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']


############################# READ DATASET #############################
def read_csv_datasets(dataset_prefixes, datasets_directory):
    """
    read the datasets in csv format and prepare them for KNN algorithm
    
    INPUT:
    -----------
    dataset_prefixes:  An array of prefixes, example: ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    datasets_directory:  is the parent directory of our datasets
    
    OUTPUTS:
    -----------
    train_ds_array:  the array of multiple train datasets
    test_ds_array:  the array of multiple test datasets
    """
    ## save the datasets into an array
    train_ds_array = []
    test_ds_array = []
    
    ## read them and append it to arrays
    for char in dataset_prefixes:
        df_train = pd.read_csv('%s%ctrain.csv' % (datasets_directory,char))
        df_test = pd.read_csv('%s%ctest.csv' % (datasets_directory,char))
        
        train_ds_array.append(df_train)
        test_ds_array.append(df_test)
    
    return train_ds_array, test_ds_array


############################# Create the confusion matrix #############################
def create_confusion_matrix(actual_labels, target_labels):
    """
    check the labels of the classifier output with actual data
    
    INPUTS:
    -----------
    actual_labels:  are the labels that the classifier gave us
    target_labels:  are the correct labels of the data
    
    OUTPUT:
    -----------
    confusion_matrix:  A Dataset with with elements of TP, TN, FP, FN as True Positives, True Negatives, False Positives and False Negative
    """
    
    a = np.array(actual_labels)
    t = np.array(target_labels)
    
    TP = ((a == True) & (t == True)).sum()
    TN = ((a == False) & (t == False)).sum()
    FP = ((a == True) & (t == False)).sum()
    FN = ((a == False) & (t == True)).sum()
 
    confusion_matrix = pd.DataFrame(data={'TP':[TP], 'TN': [TN], 'FP': [FP], 'FN': [FN]})
    
    return confusion_matrix

############################# Read the json results #############################
def read_json(filename):
    """
    read the json file of the results we made before

    INPUT:
    ------------
    filename:  the name of the file containing algorithm results

    OUTPUT:
    ------------
    dict_data:  the dictionary contaning our data
    """
    dict_data = ''
    with open(filename, 'r') as json_result:
        dict_data = json.load(json_result)
    json_result.close()
    
    return dict_data


############################# Perform the whole operation #############################
## first read the json results file
## second create a pandas dataframe
## and last write the confusion matrix
def preform_operation():
    """
    preform creating the confusion dataframe
    in the end we're creating a csv file containing confusion matrix as a dataframe
    """

    ## get the dictionary of datas we saved before
    dict_data = read_json('../KNN-Result.json')

    ds_confusion_matrix = pd.DataFrame(columns=[ 'dataset', 'K' ,'TP', 'TN', 'FP', 'FN'])

    ## just use the Atrain, Atest results
    for dataset_prefix in prefixes:

        ## get the dataset name for json file
        dataset_name = 'dataset %c' % dataset_prefix
        
        ## get the test dataframe to check the confusion matrix
        _ , test_ds = read_csv_datasets([dataset_prefix], '../../processed_dataset/')
        
        
        ## the K value in KNN algorithm
        for K in K_values:
            classified_values = []
            for j in range(0, len(dict_data[dataset_name][str(K)])):
                values = dict_data[dataset_name][str(K)][ 'index_%i_classified' % j ]
                classified_values.append(values)
                
            
            ds = create_confusion_matrix(classified_values, test_ds[0].label)
            ds['K'] = K
            ds['dataset'] = dataset_prefix
                
            ## add the confusion matrix values into the dataframe
            ds_confusion_matrix = ds_confusion_matrix.append(ds, ignore_index=True)
    
    ds_confusion_matrix.to_csv('../../processed_dataset/confusion_matrix.csv',mode='w')


if __name__ == '__main__':
    preform_operation()