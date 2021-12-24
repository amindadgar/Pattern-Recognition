import pandas as pd
import numpy as np
from memory_profiler import profile
import time
import json
import datetime

############################# DECIDE THE CLASSIFICATION RESULTS #############################
def reform_data(data, test_data ,THRESHOLD):
    """
    Reform an array of floats between one and zero with the threshold value into integers as 0 and 1
    Inputs:
    THRESHOLD is the value to set the data whether into 0 or 1 
    data is the float array
    """
    test_labels = np.array(data) >= 0.5
    test_labels = [1 if label else 0 for label in test_labels]
    classify = (test_labels == test_data.label).values
    
    return classify

############################# PERFORM KNN ALGORITHM  #############################
def KNN(K, train_data, test_data, features, threshold = 0.5):
    """
    The K nearest neighbor algorithm
    
    Inputs:
    K is the KNN algorithm parameter (a positive integer value)
    train_data and test_data in the format of pandas dataframe, the algorithm's food!
    features are the feature for each data
    threshold is the value to decide the result of KNN algorithm
    """
        
    ## get the algorithm start time
    start_time = time.time_ns()
    
    assert K > 0, 'K must be a positive value and also not zero!'

    test_label_votes = []
    
    ## start the test phase
    for index in range(0, len(test_data)):
        
        d = 0
        for feature in features:
            d += (train_data[feature]- test_data.iloc[index][feature]) ** 2
        d = np.sqrt(d)

        ## the index of K nearest neighbor points
        nearest_indexes = d.sort_values()[:K].index
                
        ## get the labels of nearest data
        labels = train_data.iloc[nearest_indexes].label
        
        ## use the mean to calculate the average vote
        vote = labels.mean()
        test_label_votes.append(vote)
        
    ## decide the voting process in KNN
    classified_data = reform_data(test_label_votes, test_data, threshold)

    finish_time = time.time_ns()
    print('Finished! K=%s algorithm time: ' % K, ((finish_time - start_time) / 1000), ' miliseconds')
        
    return classified_data

############################# READ DATASET #############################
def read_dataset():
    """
    read the datasets in csv format and prepare them for KNN algorithm
    """
    ## save the datasets into an array
    train_ds_array = []
    test_ds_array = []
    
    ## read them and append it to arrays
    for char in ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
        df_train = pd.read_csv('processed_dataset/%ctrain.csv' % char)
        df_test = pd.read_csv('processed_dataset/%ctest.csv' % char)
        
        train_ds_array.append(df_train)
        test_ds_array.append(df_test)
    
    return train_ds_array, test_ds_array

@profile
def preform_knn():
    """
    perform KNN algorithm for each dataset with different K values
    """
    
    ## We have multiple datasets with diffrent names as Atrain.csv, Btrain.csv and etc 
    dataset_prefixes = ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    
    ## read all datasets
    train_ds_array, test_ds_array = read_dataset()
    
    ## dictionary history to save all
    mdate = str(datetime.datetime.now())
    history = {'date': mdate}
    
    for i in range(0, len(train_ds_array)):
        ## save the result of different K values
        labels_classified = {}
        features = np.array(test_ds_array[i].columns[test_ds_array[i].columns != 'label'])
        print('Preforming %c Dataset (%cTrain.csv, %cTest.csv)' % (dataset_prefixes[i], dataset_prefixes[i], dataset_prefixes[i]))
        print('-----------------------------------------------')
        
        for K in [1, 2, 5, 10, 15, 20, 50, 100, 500, 1000]:
            label = KNN(K, train_ds_array[i], test_ds_array[i], features)
            labels_classified[str(K)] = {('data %i classified as' % i): str(l) for i,l in enumerate(label)}
            
            ## print the results of algorithm
            print('K=%s ' % K, 'True classified %s' % label.sum(), '\taccuracy: %f' % (label.sum()*100 / len(label)))        
                  
        ## save each dataset histories
        history[dataset_prefixes[i]] = labels_classified    
        print('\n\n')
        
    ## save the results in a file
    with open('KNN-Result.json','w') as file:
        json.dump(history, file, indent = 4)
        
if __name__ == '__main__':
    preform_knn()