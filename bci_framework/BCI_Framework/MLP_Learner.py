import numpy as np
import pickle
import sys
from Learner import Learner, NONRF_Learner
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import pairwise, zero_one_loss, mean_squared_error
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.utils import shuffle
import logging
from sklearn import cross_validation
import mlp

class MLP_Learner(NONRF_Learner):
    """applying MLP to BCI dataset"""
    
    def __init__(self, config, method = 'classification'):
        """  """
#         self.logging = logging
#         if config.configuration['logging_level_str'] == 'INFO':
#             self.logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
#         else:
#             self.logging.basicConfig(level=logging.NOTSET)
#         
#         self.logging.info('started building a MLP_Learner!') 
# 
#         self.result_path = '' 
#         self.result_opt_path = ''
        Learner.__init__(self, config, method)
        
        
#         self.logging.info('Done building a MLP_Learner instance')

    def generate_param_grid(self, feature_param_list, learner_name):
        
        learning_rate_range = [0.01, 0.1]
        L1_reg_range = [0.00, 0.1, 0.01, 0.001, 0.0001] 
        L2_reg_range  = [0.00, 0.1, 0.01, 0.001, 0.0001] 
        n_hidden_range = [5, 10, 20, 30, 40, 60, 100]
        
        if feature_param_list is None:
            scores = np.zeros(shape=(len(learning_rate_range)*len(L1_reg_range)*len(L2_reg_range)*len(n_hidden_range),
                                       self.config.configuration["number_of_cvs_dict"][learner_name]))
            param_grid = [ (l_rate, l1, l2, n_hidden) for l_rate in learning_rate_range for l1 in L1_reg_range
                            for l2 in L2_reg_range for n_hidden in n_hidden_range]
            self.grid_dictionary = {'learning_rate':0, 'l1':1, 'l2':2, 'n_hidden':3}

        else:
            scores = np.zeros(shape=(len(learning_rate_range)*len(L1_reg_range)*len(L2_reg_range)*len(n_hidden_range)* len(feature_param_list),
                                       self.config.configuration["number_of_cvs_dict"][learner_name]))
            param_grid = [ (l_rate, l1, l2, n_hidden, feat_param) for l_rate in learning_rate_range for l1 in L1_reg_range
                            for l2 in L2_reg_range for n_hidden in n_hidden_range for feat_param in feature_param_list]
            self.grid_dictionary = {'learning_rate':0, 'l1':1, 'l2':2, 'n_hidden':3, 'fe_params':4}
        return param_grid, scores
    
    def set_params_list(self, learner_params, i):
        
        learning_rate = learner_params[0]
        l1 = learner_params[1]
        l2 = learner_params[2]
        n_hidden = learner_params[3]
        
        if self.method == 'classification':
            self.learner = mlp.MLP(learning_rate, l1, l2, n_hidden)
        
        elif self.method == 'regression':
            
            sys.exit('Error! regression not implemented yet!!!!!')
           
    def set_params_dict(self, learner_params):
        
        if self.method == 'classification':
            self.learner = mlp.MLP(learner_params['learning_rate'], learner_params['l1'], 
                                   learner_params['l2'], learner_params['n_hidden'], learner_params['best_error'])
        
        elif self.method == 'regression':
            
            sys.exit('Error! regression not implemented yet!!!!!')
            
    def fit_calc_cv_scores(self, X_train, y_train, X_test, y_test):
        
        X = (X_train, X_test)
        Y = (y_train, y_test)
        self.learner.fit(X, Y)
        
        return self.predict_error(X_test, y_test)
    
#     def predict_error(self, X_test, Y_test):
# 
# 
#         preds = self.learner.predict(X_test)
#         error = np.sum((preds != Y_test))/float(len(Y_test))
#         
#         return error
    
#     def write_results_toFile(self, scores, param_grid, result_path):
#         
#         avg_errs = np.mean(scores, axis=1)
#         min_ind = np.argmin(avg_errs, axis=0)
# 
#         with open(result_path, 'w') as res_file:
#             print>>res_file, np.min(avg_errs, axis=0)
#             
#             if len(param_grid[0]) == 4:
#                 print>>res_file, dict(learning_rate = param_grid[min_ind][0], l1 = param_grid[min_ind][1],
#                                       l2 = param_grid[min_ind][2], n_hidden = param_grid[min_ind][3],
#                                       fe_params = None)
#             elif len(param_grid[0]) >= 5:
#                 print>>res_file, dict(learning_rate = param_grid[min_ind][0], l1 = param_grid[min_ind][1],
#                                       l2 = param_grid[min_ind][2], n_hidden = param_grid[min_ind][3], 
#                                       fe_params = param_grid[min_ind][4:])
#             
#             print>>res_file, np.std(scores, axis=1)
#             print>>res_file, scores   



















        
    def train_learner(self,  X, Y, X_test = [], Y_test = [], learner_params = [] ,optimal = False):
        """  """
        if optimal:
            self.train_learner_opt(X, Y, X_test, Y_test, learner_params)
        else:
            self.train_learner_cv(X, Y)
    
    def train_learner_cv(self, X, Y, optimal = False):
        """ """
        self.logging.info('Standardizing data!')
        assert self.result_path != ''
        X = np.asarray( X, dtype=np.float32, order='F')
        Y = np.asarray( Y, dtype=np.short, order='F')
        scaler = StandardScaler()

        X = scaler.fit_transform(X)
        
        self.logging.info('X size is: %s and Y size is: %s', '_'.join(map(str,X.shape)), map(str,Y.shape))

        for i in range(self.config.configuration["number_of_cvs"]):
            self.logging.info('iteration  number %s for cross validation', str(i))
            X_new, Y_new = shuffle(X, Y, random_state=i)
            cv = StratifiedKFold(y=Y_new, n_folds=self.config.configuration["number_of_cv_folds"])
            
            for param_ind in range(len(self.scores)):
                print self.param_grid[param_ind]
                accs = np.zeros(self.config.configuration["number_of_cv_folds"])
                fold_number = 0
                l_rate = self.param_grid[param_ind][0]
                l1 = self.param_grid[param_ind][1]
                l2 = self.param_grid[param_ind][2]
                n_hidden = self.param_grid[param_ind][3]
                self.learner = mlp.MLP(l_rate, l1, l2, n_hidden)    

                for train_index, test_index in cv:
#                    print("TRAIN:", train_index, "TEST:", test_index)
                    X_train, X_test = X_new[train_index], X_new[test_index]
                    y_train, y_test = Y_new[train_index], Y_new[test_index]
                    datasets = [X_train, y_train, X_test, y_test]
                    accs[fold_number] = self.learner.fit((X_train, X_test),(y_train, y_test))
#                     accs[fold_number] = mlp.apply_mlp(l_rate, l1, l2, n_hidden, datasets)
                    fold_number += 1
                    
                self.scores[param_ind,i] = np.mean(accs)

        
        min_ind = np.argmin(np.mean(self.scores, axis=1))
    
        self.logging.info('Writing the results to file!')
        with open(self.result_path, 'w') as res_file:
            print>>res_file, np.mean(self.scores)
            
            print>>res_file, dict(l_rate = self.param_grid[min_ind][0], l1 = self.param_grid[min_ind][1], 
                                  l2 = self.param_grid[min_ind][2], n_hidden = self.param_grid[min_ind][3])
            print>>res_file, np.std(self.scores, axis=1)

    def train_learner_opt(self, X, Y, X_test, Y_test, learner_params = []):
        """ """
#        self.logging.info('Standardizing data!')
#        Y_test = np.array(Y_test)
#        scaler = StandardScaler()
#        X = scaler.fit_transform(X)
#        X_test = scaler.transform(X_test)
#        self.logging.info('X size is: %s and Y size is: %s and X_test size is: %s and Y_test size is: %s',
#                           '_'.join(map(str,X.shape)), str(len(Y)), '_'.join(map(str,X_test.shape)), str(len(Y_test)))

        X, Y, X_test, Y_test = self.scale_all_data(X, Y, X_test, Y_test)
        
        if self.method == 'classification':
            clf = mlp.MLP(learner_params['l_rate'], learner_params['l1'], learner_params['l2'], learner_params['n_hidden'], learner_params['best_score'])
            self.logging.info('optimal MLP classifier trained')
        elif self.method == 'regression':
            pass
#            clf = self.learner()
#            self.logging.info('optimal linear regressor trained with alpha = %s!', str(learner_params['C']))
        
        clf.fit((X, X_test), (np.array(Y), np.array(Y_test)))
        
        self.fit_opt_learner(X, Y, X_test, Y_test, clf)
        #        clf.fit(X, Y)
#
#        Y_pred_train = clf.predict(X)
#        
#        Y_pred = clf.predict(X_test)
#        nonnan_indices = ~np.isnan(Y_test)
#        error = self.my_loss(Y_test[nonnan_indices], Y_pred[nonnan_indices])
#        self.logging.info('error is %s', str(error))
#
#        probs_train = clf.predict_proba(X)
#        probs_test = clf.predict_proba(X_test)
#
#        np.savez(self.result_opt_path, error=error, Y_pred=Y_pred, Y_pred_train=Y_pred_train, probs_train=probs_train, probs_test = probs_test)
#        with open(self.result_opt_path,'w') as res_file:
#            res_file.write(str(error))
#            res_file.write('\n')
#            res_file.write(' '.join(map(str, Y_pred) + ['\n']))
#            res_file.write('\n')
#            res_file.write(' '.join(map(str, Y_pred_train) + ['\n']))
#            res_file.write('#Train probabilities: ')
#            res_file.write(' '.join(map(str, probs) + ['\n']))
#
#            res_file.write('#Test probabilities: ')
#            res_file.write(' '.join(map(str, probs_test) + ['\n']))

        
