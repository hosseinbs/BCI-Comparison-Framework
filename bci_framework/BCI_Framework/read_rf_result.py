from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import sys
import os

slash = '\\'
data_dir = 'E:\Users\hossein\data_bci_rf'
results_dir = 'E:\Users\hossein\data_bci_rf_resutls1'
class_labels = [-1, 0, 1]
all_subjects_oob_predictions = {'a':{},'b':{},'g':{},'f':{}}
all_subjects_inbag_predictions = {'a':{},'b':{},'g':{},'f':{}}
all_subjects_oob_scores = {'a':{},'b':{},'g':{},'f':{}}
all_subjects_inbag_scores = {'a':{},'b':{},'g':{},'f':{}}

def main():
    
    interval_dirs = os.listdir(data_dir)

    for interval_dir in interval_dirs:
        
        result_interval_dir = results_dir + slash + interval_dir
        if not os.path.exists(result_interval_dir):
            os.makedirs(result_interval_dir)
        
        window_dirs = os.listdir(data_dir + slash + interval_dir)
        
        for window_dir in window_dirs:
            
            result_window_dir = result_interval_dir + slash + window_dir
            if not os.path.exists(result_window_dir):
                os.makedirs(result_window_dir)
        
        
            current_data_dir = data_dir + slash + interval_dir + slash + window_dir
            current_results_dir = results_dir + slash + interval_dir + slash + window_dir

            for files in os.listdir(current_data_dir):

                if files.endswith("matX.txt"):
                    subject = files[files.find('data') + 4]
                    y_file = current_data_dir + slash + files.replace('matX', 'matY')
                    x_file = current_data_dir + slash + files
                    subject_file = current_results_dir + slash + subject
                    if os.path.isfile(subject_file):
                        oob_predictions, inbag_predictions, oob_scores, inbag_scores = calc_OOB_inbag_scores(x_file, y_file, subject_file)
                    
                        all_subjects_oob_predictions[subject].update({(interval_dir,window_dir):oob_predictions})
                        all_subjects_inbag_predictions[subject].update({(interval_dir,window_dir):inbag_predictions})
                        all_subjects_oob_scores[subject].update({(interval_dir,window_dir):oob_scores})
                        all_subjects_inbag_scores[subject].update({(interval_dir,window_dir):inbag_scores})
                        print interval_dir + " : " + window_dir + " : " +subject

                        
        
        write_results()
        ### use dictionary to save results
                        

def calc_OOB_inbag_scores(x_file, y_file, subject_file):
    
    X = np.loadtxt(x_file)
    X = X.T
    Y = np.loadtxt(y_file)
                    
    pkl_file = open(subject_file, 'r')
    current_forest = pickle.load(pkl_file)
    oob_predictions = np.zeros([3,len(Y)])
    inbag_predictions = np.zeros([3,len(Y)])
    
    oob_scores = np.zeros(len(current_forest.estimators_))
    inbag_scores = np.zeros(len(current_forest.estimators_))
    indices = np.array(range(len(Y)))

    for index in range(len(current_forest.estimators_)):
        d3 = current_forest.estimators_[index]
        current_tree_predictions = d3.predict(X)
        
        oob_predictions = calc_current_tree_prediction(oob_predictions,current_tree_predictions,~d3.indices_,indices) ## OOB scores
        inbag_predictions = calc_current_tree_prediction(inbag_predictions,current_tree_predictions,d3.indices_,indices) ## inbag scores
        
    
        oob_scores[index] = calc_scores(oob_predictions,Y) ## calc oob scores
        inbag_scores[index] = calc_scores(inbag_predictions,Y) ## calc oob scores
    
    
#    pylab.plot(oob_scores)
#    pylab.show()
    pkl_file.close()
    return oob_predictions, inbag_predictions, oob_scores, inbag_scores


def calc_scores( given_predictions, Y):
    
    all_predictions = np.argmax(given_predictions,0)
    all_predictions -= 1 
      
    number_of_matches = (all_predictions == Y).tolist().count(True)
    return float(number_of_matches)/len(Y)  

    
def calc_current_tree_prediction( sample_predictions, current_tree_predictions, given_indices, all_indices):
    
    current_samples_predictions = current_tree_predictions[given_indices]
        
    current_samples_indices = all_indices[given_indices]
    
    for index in range(len(class_labels)):
        label = class_labels[index]
        sample_predictions[index,current_samples_indices[current_samples_predictions == label]] += 1
    
    return sample_predictions



def write_results():
    
    pkl_file = open('all_results', 'w')
#    with open("test.txt", "a") as myfile:
#        myfile.write("appended text")

    my_list = [all_subjects_oob_predictions, all_subjects_inbag_predictions, all_subjects_oob_scores, all_subjects_inbag_scores]
    
    pickle.dump(my_list, pkl_file)
    
    pkl_file.close()




main()