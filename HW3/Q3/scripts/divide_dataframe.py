import pandas as pd
import sys

def divide_dataframe(dataset, validation_split, label_string):
    """
    Divide a pandas dataframe into two dataframes

    INPUTS:
    ---------
    dataset:  the pandas dataframe that is going to be splited
    validation_split:  the partition of how the validation set will be, example: 0.1
    label:  For a multiclass classification, we need the label to get sample from each label (Must be an string)

    OUTPUTS:
    ---------
    train_set:  pandas dataframe, the partition of training set
    valid_set:  pandas dataframe, the partition of validation set  
    """

    assert ((validation_split >0) & (validation_split < 1)), "[ERROR] validation_split must be between 0 and 1!"
    columns = dataset.columns
    train_images = pd.DataFrame(columns=columns)
    validation_images = pd.DataFrame(columns=columns)

    ## get all labels for training and test data
    for label in dataset[label_string].unique():
        ## get the dataset for each dataset label
        dataset_label = dataset[dataset[label_string] == label]
        length = int(len(dataset_label) * validation_split)

        train_images = train_images.append(dataset_label.iloc[: length], ignore_index=True)
        validation_images = validation_images.append(dataset_label.iloc[length: ], ignore_index=True)

    features = columns[:len(columns) - 1]

    ## normalize the values into float
    ## we need to convert the integer values to float for KNN function
    for col in features:
        train_images[col] = pd.to_numeric(train_images[col], downcast='float')
        validation_images[col] = pd.to_numeric(validation_images[col], downcast='float')

    return train_images, validation_images


if __name__ == '__main__':
    """ get the arguments
     these arguments must be given
    
     dataset_name in string format
     validation_split a number between 0 and 1
     label in string format 
    """
    arguments = sys.argv

    assert len(arguments) == 4, f"[ERROR] 3 arguments must be added! input arguments is {len(arguments)}"

    dataset_name = arguments[1]
    validation_split = float(arguments[2])
    label = str(arguments[3])

    dataframe = pd.read_csv(dataset_name)

    train_df, validation_df = divide_dataframe(dataframe, validation_split, label)

    ## TODO: return the dataframes to another file
