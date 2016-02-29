BCI Framework
_______________

BCI Framework is a package to compare different classification 
and feature extraction algorithms in BCI:

Please see the paper:
"Comparing different classifiers in Sensory Motor Brain Computer Interfaces". Hossein Bashashati, Rabab K. Ward, Gary E. Birch, Ali Bashashati

The code is designed compare differnet algorithms, easy to extend and can replicate the exact results in the paper. 

Dependencies
_______________

Python 2.7

* Numpy
* Scipy
* Scikit-learn 0.15
* Theano
* Matplotlib
* spectrum

This software has been tested on linux and Windows.

How to tun the program
_______________________

To run the program, run the following command in git terminal:

git clone git://github.com/hosseinbs/bci_framework.git

Then run the following command: 

/bci_framework/run_bcic.py dataset_name classification_method Feature_extraction number_of_CSPs channels train_or_test

*Dataset_name = "BCICIII3b", "BCICIV2a", "BCICIV2b", "BCICIV1", "SM2"

*Classification_method = 'MLP', 'LDA', 'QDA', 'RANDOM_FOREST', 'SVM', 'LogisticRegression', 'Boosting'

*Feature_extraction = 'BP', 'logbp', 'morlet'

*Number_of_CSPs = '-1' in case of not using CSP, otherwise an integer 

*Channels = 'ALL', 'CS', 'C34', 'C34Z', 'CSP'

*Train_or_test = 'run', 'eval' -- the program should first be run with the 'run' argument, 
then to evaluate the results on the test data 'eval' should be used


Replication of the results of the framework paper
_________________________________________________

The scripts to replicate the results of the framework paper is in folder 'Batch_files'


Run on other datasets
_____________________

To run the algorithms on other datasets, copy your data to the folder 'bci_data'. 

Use the same format as the files in the folder. Create a folder with the name your_dataset_name. Create two folder 'TrainData' and 
'TestData' in that folder. Add a file with the name 'yourDatasetName_spec' to the folder 'bci_framework/BCI_Framework'. 
Use the files 'BCICIII3b_spec', 'BCICIV2a_spec' , ... as examples.

Format of the data:
Each row in subject_X.txt is a feature vector and each row in subject_Y.txt is its corresponding label.

Copy the training files in TrainData folder and test files in TestData folder.


Extending the software
______________________

The software is designed to be modular. The main file of this BCI_Framework/Main.py, the function Main.run_learner_gridsearch() performs grid search on the 

hyper-parameters which are given in file 'yourDatasetName_spec'. the function Main.submit_train_learner() either submits the jobs to a cluster (which performs grid search in parallel)

(POSIX case) or runs the code on a single machine (nt case).

Class 'BCI_Framework/Configuration_BCI' reads the 'yourDatasetName_spec' file.

To add a new classification algorithm you should extend the "NONRF_Learner" in Learner.py.

To add a new feature extraction algorithm you should add the function in Feature_Extractor.py.

____________________________________________________________________________________________________________

If you have any questions about this software please feel free to contact 'hosseinbs@ece.ubc.ca'


