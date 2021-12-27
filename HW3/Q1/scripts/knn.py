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
    
    INPUTS:
    -----------
    THRESHOLD:  is the value to set the data into False or True
    data:  is the float array
    
    OUTPUT:
    -----------
    test_labels:  the array showing that the data tends to which class 
    """
    test_labels = np.array(data) >= 0.5
    
    return test_labels

############################# PERFORM KNN ALGORITHM  #############################
def KNN(K, train_data, test_data, features, threshold = 0.5):
    """
    The K nearest neighbor algorithm
    
    INPUTS:
    -----------
    K:  is the KNN algorithm parameter (a positive integer value)
    train_data, test_data:  in the format of pandas dataframe, the algorithm's food!
    features:  are the feature for each data
    threshold:  is the value to decide the result of KNN algorithm
    
    OUTPUT:
    -----------
    classified_data:  showing the classifier results
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
def read_dataset(datasets_directory):
    """
    read the datasets in csv format and prepare them for KNN algorithm
    
    INPUTS:
    -----------
    datasets_directory:  the parent directory for our datasets 

    OUTPUTS:
    -----------
    train_ds_array:  the array of multiple train datasets
    test_ds_array:  the array of multiple test datasets
    """
    ## save the datasets into an array
    train_ds_array = []
    test_ds_array = []
    
    ## read them and append it to arrays
    for char in ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
        # df_train = pd.read_csv('processed_dataset/%ctrain.csv' % char)
        # df_test = pd.read_csv('processed_dataset/%ctest.csv' % char)

        train_dir = '%s%ctrain.csv' % (datasets_directory, char)
        test_dir = '%s%ctest.csv' % (datasets_directory, char)
        df_train = pd.read_csv(train_dir)
        df_test = pd.read_csv(test_dir)
 
        
        train_ds_array.append(df_train)
        test_ds_array.append(df_test)
    
    return train_ds_array, test_ds_array


############################# check labels of the classified data #############################
def check_label(actual_labels, target_labels):
    """
    check the labels of the classifier output with actual data
    
    INPUTS:
    -----------
    actual_labels:  are the labels that the classifier gave us
    target_labels:w  are the correct labels of the data
    
    OUTPUT:
    -----------
    equal_labels:  showing which data the classifier, classified correct and wrong
    """
    
    ## convert to numpy arrays to compute the equality of each elemnts
    a_labels = np.array(actual_labels)
    t_labels = np.array(target_labels)
    
    ## mark as True the eqaul labels
    equal_labels = (a_labels == t_labels)
    
    return equal_labels
    

@profile
def preform_knn():
    """
    perform KNN algorithm for each dataset with different K values
    
    And at the end writing the results in a file
    """
    
    ## We have multiple datasets with diffrent names as Atrain.csv, Btrain.csv and etc 
    dataset_prefixes = ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    
    ## read all datasets
    train_ds_array, test_ds_array = read_dataset('../processed_dataset/')
    
    ## dictionary history to save all
    mdate = str(datetime.datetime.now())
    history = {'knn.py script run date': mdate}
    
    for i in range(0, len(train_ds_array)):
        ## save the result of different K values
        labels_classified = {}
        features = np.array(test_ds_array[i].columns[test_ds_array[i].columns != 'label'])
        print('Preforming %c Dataset (%cTrain.csv, %cTest.csv)' % (dataset_prefixes[i], dataset_prefixes[i], dataset_prefixes[i]))
        print('-----------------------------------------------')
        
        for K in [1, 2, 5, 10, 15, 20, 50, 100, 500, 1000]:
            label = KNN(K, train_ds_array[i], test_ds_array[i], features)
            labels_classified[str(K)] = {('index_%i_classified' % i): int(l) for i,l in enumerate(label)}
            
            ## check correct classification results
            result = check_label(label, test_ds_array[i].label)
            
            ## print the results of algorithm
            print('K=%s ' % K, 'correctly classified count: %s' % result.sum(), '\taccuracy: %f' % (result.sum()*100 / len(result)))        
                  
        ## save each dataset histories
        history['dataset %c' % dataset_prefixes[i]] = labels_classified    
        print('\n\n')
        
    ## save the results in a file
    with open('../KNN-Result.json','w') as file:
        json.dump(history, file, indent = 4)
    file.close()
        
if __name__ == '__main__':
    preform_knn()