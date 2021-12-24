import numpy as np
import pandas as pd

############################# process dataset data  #############################
def process_string_dataset(string_data, delimiter = ' '):
    """
    convert the data we have in string to float
    string_data can be lines of strings
    """
    ## count of data lines
    data_line_count = len(string_data)

    ## Data is our data values in float
    data_floats = []
    ## create a temp string to save each numbers character
    temp_string = ''

    ## go through the data and convert it to float
    for index in range(0, data_line_count):
        data_floats.append([])
        for character in string_data[index][1:]:
            if(character != delimiter and character != '\n'):
                temp_string += character
            else:
                ## if the string goes to delimiter convert the data into float
                ## and reset temp_string variable to save the next data
                data_floats[index].append(float(temp_string))
                temp_string = ''
                
    return data_floats


############################# read dataset txt file #############################
def read_dataset(name):
    """
    read our given dataset and return the datas
    name is the dataset directory with its name can be forexample: ../toy/Atrain
    """
    file = open(name, 'r')

    lines = file.readlines()
    data_lines = []
    for line in lines:
        ## dataset with # in the start does not contain data
        if(line[0] != '#'):
            data_lines.append(line)
            
    ## convert to float and return the ready dataset
    float_data = process_string_dataset(data_lines)
    return float_data



############################# attach different class dataframes #############################
def attach_classes(data_frames, labels):
    """
    Consider each dataframe a class of our data
    attach the dataframes into one and add the class lebels
    Inputs:
    data_frames is an array of data_frames
    lebels is the labels to the classes
    """
    
    data_frame_count = len(data_frames)
    
    ## create an empty dataframe
    df = pd.DataFrame()
    
    for i in range(0, data_frame_count):
        ## create the label array for each class with the class data count
        class_label = np.full(len(data_frames[i]), labels[i])
        ## add the label to each class
        data_frames[i]['label'] = class_label 
        df = df.append(data_frames[i], ignore_index=True)
        
    return df

############################# convert datasetmatrix to pandas dataframe #############################
def convert_to_pd_dataframe(matrix_data):
    """
    convert a dataset in the form of matrix into pandas dataframe
    Inputs:
    matrix_data with two feature space
    """
    df = pd.DataFrame(matrix_data, columns=['feature_one', 'feature_two']) 
    
    return df

############################# the whole converting raw data into pandas dataframe #############################
def convert_rowData_df(row_Datas, labels):
    """
    convert raw data into dataframes
    Input:
    row_Datas must be an array with the raw datas,
    if we had one raw data then it must be as [raw_data]
    each raw_data is for one class
    
    lebels is the labels to the data classes
    """
    ## create an array containing the dataframes 
    df_class_arrays = []
    
    ## get the count of row datas needed to be converted
    count = len(row_Datas)
    
    ## go through each row data array and create get a dataframe for each
    for i in range(0, count):
        df = convert_to_pd_dataframe(np.matrix(row_Datas[i]))
        df_class_arrays.append(df)
    
    ## attach the dataframes of each row data (each row data represent a class)
    df = attach_classes(df_class_arrays, labels)
    
    return df
        
############################# Main function doing all the work #############################
def main():
    """
    Do all the works
    Read txt datasets and convert them into multiple csv files
    """
    ## read txt datasets 
    for char in ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
        train_raw_ds = read_dataset('toy dataset/%ctrain' % char)
        train_class1 = np.matrix(train_raw_ds[:2])
        train_class2 = np.matrix(train_raw_ds[2:])
        
        df_train = convert_rowData_df([train_class1.T, train_class2.T], [0, 1])
        
        df_train.to_csv('processed_dataset/%ctrain.csv' % char, index=False, mode='w')
        
        test_raw_ds = read_dataset('toy dataset/%ctest' % char)
        ## get classes and convert to matrix representation
        test_raw_class1 = np.array(test_raw_ds).T[1000:]
        test_raw_class2 = np.array(test_raw_ds).T[:1000]
        
        test_raw_class1 = np.matrix(test_raw_class1)
        test_raw_class2 = np.matrix(test_raw_class2)

        df_test = convert_rowData_df([test_raw_class1, test_raw_class2], [0, 1])
        df_test.to_csv('processed_dataset/%ctest.csv' % char,index=False)
        

        
if __name__ == '__main__':
    main()