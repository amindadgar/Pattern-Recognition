import numpy as np
import pandas as pd

def check_confusion_matrix(actual_label, target_label):
    """
    create the confusion matrix and accuracy and insert them into a pandas dataframe 

    INPUTS:
    ---------
    actual_label:  the label classified by the classifier
    target_label:  the right labels for data from dataset

    OUTPUT:
    -----------
    confusion_matrix:  A Dataset with with elements of TP, TN, FP, FN, accuarcy as True Positives, True Negatives, False Positives, False Negative, and classifier accuracy
    """
    
    a = np.array(actual_label)
    t = np.array(target_label)
    
    TP = ((a == True) & (t == True)).sum()
    TN = ((a == False) & (t == False)).sum()
    FP = ((a == True) & (t == False)).sum()
    FN = ((a == False) & (t == True)).sum()
 
    confusion_matrix = pd.DataFrame(data={'TP':[TP], 'TN': [TN], 'FP': [FP], 'FN': [FN]})
    confusion_matrix['accuracy'] = (a == t).sum() * 100 / len(actual_label)
    
    return confusion_matrix

def bayes_classifier(datasets_name, mu_s, Sigma_s):
    """
    use the bayes classifier for multiple datasets

    INPUTS:
    --------
    datasets:  array of multiple datasets name
    mu_s:  [mu1, mu2] multidimensional array, array of mean for class1 and class2 vectors for each dataset (Note each mu shape is 2*1)
    Sigma_s:  [Sigma1, Sigma2] multidimensional array, array of covariance for class1 and class2 vectors for each dataset

    OUTPUT:
    --------
    ds_results:  pandas dataframe of results containing dataset_name,confusion_matrix and accuracy
    """

    ## check the values are correctly entered
    assert len(datasets_name) == len(mu_s), 'Error! not enough mean array for datasets was given!\ndatasets array length must be equal to mean (mu) array length '
    assert len(Sigma_s) == len(mu_s), 'Error! not enough Sigma data for mu datas was given!\ncovariance (Sigma) array length must be equal to mean array length '
    assert len(Sigma_s) == len(datasets_name), 'Error! not enough Sigma data for datasets was given!\ncovariance (Sigma) array length must be equal to dataset array length'

    ## check all the parameters for each class was entered correctly!
    ## the Sigma_s or the covariance array has the shape of 6 arrays and each array consist of two matrixes with 4 element (2 by 2 matrix) 
    assert len(Sigma_s.flatten()) == len(Sigma_s) * 2 * 4, 'Error! covariances for each class was not entered correctly!\nSigma_s shape is 6 arrays of two matrixes with 4 element (2 by 2 matrix)'
    assert len(mu_s.flatten()) == len(mu_s) * 2 * 2, 'Error! mean for each class was not entered!\nmu_s must be an array of six elements each element of contains a vector by size of 2*1!'


    ds_results = pd.DataFrame(columns=[ 'dataset', 'TP', 'TN', 'FP', 'FN', 'accuracy'])
    
    ## iterate over each dataset
    for i in range(0, len(datasets_name)):

        ## to save each label
        classified_labels = []
        ## read dataset
        dataset = pd.read_csv(datasets_name[i])
        
        ## iterate to get each row in dataset
        for j in range(0, len(dataset)):
            
            ## get each feature for each row
            x1 = ds_A['feature_one'].iloc[j]
            x2 = ds_A['feature_two'].iloc[j]

            ## calculate the probability for calss 1
            p_class1 = guassian_multivariate(mu_s[i][0], Sigma_s[i][0], x1, x2)

            ## calculate the probability for calss 2
            p_class2 = guassian_multivariate(mu_s[i][1], Sigma_s[i][1], x1, x2)

            ## class 2 has the label 1 (True)
            ## class 1 has the label 0 (False)
            label = p_class2 > p_class1

            classified_labels.append(label)
        
        ## result of one of the datasets
        ds_partial_result = pd.DataFrame()
        ds_partial_result = check_confusion_matrix(classified_labels, dataset.label)
        ds_partial_result['dataset'] = datasets_name[i][-10:]

        ds_results = ds_results.append(ds_partial_result, ignore_index=True)

    return ds_results
    
def perform_classifying():
    ## create the mean vectors array for each dataset A, B, C, D, E, F
    mu_s = np.array([[[0,0], [2, 2]], 
                [[0,0], [0,0]], 
                [[0,0], [0,0]],
                [[0,0], [0,0]], 
                [[0,0], [-0.8,0.8]],
                [[0,0], [1,6]]])

    ## create the covariance matrix array for each dataset A, B, C, D, E, F
    ## each class have different covariance
    Sigma_A1 = np.matrix('1 -0.8; -0.8 1')
    Sigma_A2 = np.matrix('1 0.8; 0.8 1')

    Sigma_B1 = np.matrix('1 -0.75; -0.75 1')
    Sigma_B2 = np.matrix('1 0.75; 0.75 1')

    Sigma_C1 = np.matrix('1.75 0; 0 0.25')
    Sigma_C2 = np.matrix('0.25 0; 0 1.75')

    Sigma_D1 = np.matrix('1 0; 0 1')
    Sigma_D2 = np.matrix('9 0; 0 9')

    Sigma_E1 = np.matrix('3 1; 1 0.5')
    Sigma_E2 = np.matrix('3 1; 1 0.5')

    Sigma_F1 = np.matrix('3 1; 1 0.5')
    Sigma_F2 = np.matrix('3 1; 1 0.5')

    ## appand all into an array
    Sigma_s = np.array([[Sigma_A1, Sigma_A2],
                        [Sigma_B1, Sigma_B2],
                        [Sigma_C1, Sigma_C2],
                        [Sigma_D1, Sigma_D2],
                        [Sigma_E1, Sigma_E2],
                        [Sigma_F1, Sigma_F2] ])

    dataset_names = []
    for char in ['A', 'B', 'C', 'D', 'E', 'F']:
        dataset_names.append(f"../../processed_dataset/{char}train.csv")

    result_ds = bayes_classifier(dataset_names, mu_s, Sigma_s)

    result_ds.to_csv('Q2_p1_out.csv')
    
    print('Result saved in Q2_p1_out.csv')