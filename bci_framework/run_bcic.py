import os
import sys
os.chdir('../bci_framework')
sys.path.append('./BCI_Framework')

import Main
import Single_Job_runner as SJR
import numpy as np


if __name__ == '__main__':

    myPython_path = 'python'
        
    print sys.argv
    
    dataset_name = sys.argv[1]
    if not (dataset_name == 'BCICIII3b' or dataset_name == 'BCICIV1' or dataset_name == 'BCICIV2a' or dataset_name == 'BCICIV2b' or dataset_name == 'SM2'):
        print "Dataset name == incorrect"
        sys.exit()
    
    classifier = sys.argv[2]
    if not (classifier == 'MLP' or classifier == 'LDA' or classifier == 'QDA' or classifier == 'RANDOM_FOREST' or
            classifier == 'SVM' or classifier == 'LogisticRegression' or classifier == 'Boosting'):
        print "Classifier name is incorrect"
        sys.exit()
    
    feature = sys.argv[3]
    if not (feature == 'wackerman' or feature == 'BP' or feature == 'logbp' or feature == 'morlet' or feature == 'AR'):
        print "feature name is incorrect"
        sys.exit()
    
    number_of_CSPs = int(sys.argv[4])
    channels = sys.argv[5]
    train_or_test = sys.argv[6].rstrip()

    print dataset_name, classifier, feature, number_of_CSPs
######################################################################MLP#############################################
     
    ##### perform grid search to find optimal parameters 
    print sys.argv
    if train_or_test == 'run':
        print "run"
        bcic = Main.Main('BCI_Framework', dataset_name, classifier, feature, channels, number_of_CSPs, myPython_path)
        bcic.write_feature_matrices_gridsearch()
        # bcic.run_learner_gridsearch()
    else:
        print "eval"
        bcic = Main.Main('BCI_Framework', dataset_name, classifier, feature)
        bcic.test_learner()
        
