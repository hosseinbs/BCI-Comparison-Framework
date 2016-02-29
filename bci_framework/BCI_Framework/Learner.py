from sklearn.metrics import zero_one_loss, mean_squared_error
import logging
from sklearn.preprocessing import StandardScaler
import numpy as np
import cPickle
from sklearn.metrics import precision_recall_fscore_support
import json

class Learner(object):

    def __init__(self, config, method = 'classification'):
        
        self.logging = logging
        if config.configuration['logging_level_str'] == 'INFO':
            self.logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
        else:
            self.logging.basicConfig(level=logging.NOTSET)
        
        self.logging.info('started building a learner!') 
        self.config = config
        self.method = method
        
        self.scoring = self.config.configuration['cost_function_str']
        if self.scoring == 'zero_one':
            self.test_phase_loss = zero_one_loss
        elif self.scoring == 'fp_rate_fn_rate':
            self.test_phase_loss = self.false_positive_false_negative_rates 
        elif self.scoring == 'mse':
            self.test_phase_loss = mean_squared_error
#             self.scoring = "mse"
        
        self.grid_dictionary = None
        self.logging.info('A %s Learner object is built! with scoring: %s ', method, self.scoring)
    
   
    def fit(self, X_train, y_train):
        
        if hasattr(self, 'type'):
            if self.type == 'QDA' and self.config.configuration['dataset_str'] == 'BCICIV2a':
                n_samples_for_each_class = np.bincount(y_train)
                final_n_samples_for_each_class = np.min(n_samples_for_each_class[1:])
                n_classes = len(n_samples_for_each_class) - 1
                to_be_kept_sample_indices = np.zeros(n_classes*final_n_samples_for_each_class)
                start_ind = 0
                for lebel_ind, label in enumerate(set(y_train)):
                    temp = np.where(y_train== label)[0] 
                    to_be_kept_sample_indices[start_ind:start_ind+final_n_samples_for_each_class] = temp[0:final_n_samples_for_each_class]
                    start_ind += final_n_samples_for_each_class
                
                X_train = X_train[map(int, to_be_kept_sample_indices),:]
                y_train = np.array(y_train)[map(int, to_be_kept_sample_indices)]
                
        
        self.learner.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.learner.predict(X_test)

    def false_positive_false_negative_rates(self, y_true, y_predicted):
        """ Assuming ones are positive samples and zeros are negative samples"""
        negative_indices = np.where(y_true==0)
        fp = np.sum(y_predicted[negative_indices[0]])/float(len(negative_indices[0]))
        
        positive_indices = np.where(y_true==1)
        fn = np.sum(y_predicted[positive_indices[0]]==0)/float(len(positive_indices[0]))
        
        error = zero_one_loss(y_true, y_predicted)*100.0
        
        return [fp, fn, error]
     
    
class NONRF_Learner(Learner):
    
    def fit_calc_cv_scores(self, X_train, y_train, X_test, y_test):
        
        self.learner.fit(X_train, y_train)
        return self.predict_error(X_test, y_test)
    
    def predict_error(self, X_test, Y_test):

        preds = self.learner.predict(X_test)
        classification_error = np.sum((preds != Y_test))/float(len(Y_test))
        precision, recall, _ , _ = precision_recall_fscore_support(Y_test, preds, average='weighted')
        
        return classification_error, precision, recall

    def write_cv_results_toFile(self, scores, precision_scores, recall_scores, param_grid, result_path):
        
        final_results_dict = {}
        all_scores_dict = {'error' : scores, 'precision' : precision_scores, 'recall' : recall_scores}
        for score in all_scores_dict:
            avg_score = np.mean(all_scores_dict[score], axis=1)
            if score == 'error':
                opt_ind = np.argmin(avg_score, axis=0)
            else:
                opt_ind = np.argmax(avg_score, axis=0)
            
            final_results_dict[score] = avg_score[opt_ind]
            final_results_dict[score + '_std'] = np.std(scores[opt_ind,:])
            params_dict = {}
            for element in self.grid_dictionary:
                params_dict[element] =  param_grid[opt_ind][self.grid_dictionary[element]]
            
            final_results_dict[score + '_params'] = params_dict
            if not 'fe_params' in params_dict:
                final_results_dict[score + '_params'].update(dict(fe_params = None))
           
        final_results_dict['channel_type'] = self.config.channel_type
        json.dump(final_results_dict, open(result_path,'w'))  
