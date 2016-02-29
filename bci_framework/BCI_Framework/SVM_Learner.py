import numpy as np
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
import pickle
import Error
from sklearn.metrics import pairwise, zero_one_loss, mean_squared_error
from Learner import Learner,NONRF_Learner

class SVM_Learner(NONRF_Learner):
    
    def __init__(self, config, method = 'classification'):
        
        Learner.__init__(self, config, method)
        self.kernel = 'rbf' 
        self.grid_dictionary = None
#         self.kernel = kernel_type
#         self.C_range = 10.0 ** np.arange(-2, 9)
#         self.gamma_range =  10.0 ** np.arange(-5, 4)#np.logspace(-5, -1, 10)#
#         self.degree_range = np.arange(1, 6)
#         
#         if self.kernel == 'rbf':
#             self.scores = np.zeros(shape=(len(self.C_range)*len(self.gamma_range), self.config.configuration["number_of_cvs"]))
#             self.param_grid = dict(gamma=self.gamma_range, C=self.C_range)
#         
#         elif self.kernel == 'poly':
#             self.scores = np.zeros(shape=(len(self.C_range)*len(self.degree_range), self.config.configuration["number_of_cvs"]))
#             self.param_grid = dict(degree=self.degree_range, C=self.C_range)
#         
#         elif self.kernel == 'linear':
#             self.scores = np.zeros(shape=(len(self.C_range),self.config.configuration["number_of_cvs"]))
#             self.param_grid = dict(C=self.C_range)
#         
#         elif self.kernel == 'mykernel':
#             self.scores = np.zeros(shape=(len(self.C_range)*len(self.gamma_range), self.config.configuration["number_of_cvs"]))
#             self.param_grid = [ (c,gamma) for c in self.C_range for gamma in self.gamma_range ]
#         
#         
#         if method == 'classification':
#             self.learner = SVC
#         elif method == 'regression':
#             self.learner = SVR

    def generate_param_grid(self, feature_param_list, learner_name):
        
#         kernel_types = ['rbf', 'linear']
        C_range = 10.0 ** np.arange(-2, 7)
        gamma_range =  10.0 ** np.arange(-5, 4)

        if feature_param_list is None:
            scores = np.zeros(shape=(len(C_range)*len(gamma_range), self.config.configuration["number_of_cvs_dict"][learner_name]))
            param_grid = [ (c,gamma) for c in C_range for gamma in gamma_range ]
            self.grid_dictionary = {'C':0, 'gamma':1}
        else:
            scores = np.zeros(shape=(len(C_range)*len(gamma_range)* len(feature_param_list),
                                      self.config.configuration["number_of_cvs_dict"][learner_name]))
            param_grid = [ (c, gamma, feat_param) for c in C_range for gamma in gamma_range 
                          for feat_param in feature_param_list]
            self.grid_dictionary = {'C':0, 'gamma':1, 'fe_params':2}
            
        return param_grid, scores
    
    def set_params_list(self, learner_params, i):
        
        c = learner_params[0]
        gamma = learner_params[1]
        
        if self.method == 'classification':
            self.learner = SVC(kernel = self.kernel, cache_size=2000, 
                               C = c, gamma = gamma, probability = True)
        
        elif self.method == 'regression':
            self.learner = SVR(kernel = self.kernel, cache_size=2000, 
                               C = c, gamma = gamma, probability = True)
           
    def set_params_dict(self, learner_params):
        
        default_tolerance = 0.001
        if learner_params['C'] >= 10^7:
            default_tolerance = 0.005
        if self.method == 'classification':
            self.learner = SVC(kernel = self.kernel, cache_size=2000, 
                               C = learner_params['C'], gamma = learner_params['gamma'], probability = True, tol = default_tolerance)
        
        elif self.method == 'regression':
            self.learner = SVR(kernel = self.kernel, cache_size=2000, 
                               C = learner_params['C'], gamma = learner_params['gamma'], probability = True, tol = default_tolerance)
            
            
#     def fit_calc_cv_error(self, X_train, y_train, X_test, y_test):
#         
#         self.learner.fit(X_train, y_train)
#         return self.predict_error(X_test, y_test)
#     
#     def predict_error(self, X_test, Y_test):
# 
#         preds = self.learner.predict(X_test)
#         error = np.sum((preds != Y_test))/float(len(Y_test))
#         
#         return error
    
#     def write_results_toFile(self, scores, param_grid, result_path):
#         
#         avg_errs = np.mean(scores, axis=1)
# 
#         min_ind = np.argmin(avg_errs, axis=0)
# 
#         with open(result_path, 'w') as res_file:
#             print>>res_file, np.min(avg_errs, axis=0)
#             
#             if len(param_grid[0]) == 2:
#                 print>>res_file, dict(C = param_grid[min_ind][0], gamma = param_grid[min_ind][1],
#                                    fe_params = None)
#             elif len(param_grid[0]) >= 3:
#                 print>>res_file, dict(C = param_grid[min_ind][0], gamma = param_grid[min_ind][1]
#                                    , fe_params = param_grid[min_ind][2:])
#             
#             print>>res_file, np.std(scores, axis=1)
#             print>>res_file, scores   











    def my_kernel(self, x, y, gamma):
        """ """
        
        gram = pairwise.rbf_kernel(x, y, gamma)
        gram = np.dot(gram, np.exp(-2))
        return gram
        
    def train_learner(self,  X, Y, X_test = [], Y_test = [], learner_params = [] ,optimal = False):
        
        if self.kernel != 'mykernel':
            if optimal:
                self.train_learner_opt(X, Y, X_test, Y_test, learner_params)
            else:
                self.train_learner_cv(X, Y)
        else:
            if optimal:
                self.train_learner_opt_mykernel(X, Y, X_test, Y_test, learner_params)
            else:
                self.train_learner_cv_mykernel(X, Y)
    
    def train_learner_cv_mykernel(self, X, Y):
        """ """
        scaler = StandardScaler()

        X = scaler.fit_transform(X)
        Y = np.array(Y)
        for i in range(self.config.configuration["number_of_cvs"]):
            X_new, Y_new = shuffle(X, Y, random_state=i)
            cv = StratifiedKFold(y=Y_new, n_folds=self.config.configuration["number_of_cv_folds"])
            
            for param_ind in range(len(self.scores)):
                print self.param_grid[param_ind]
                accs = np.zeros(self.config.configuration["number_of_cv_folds"])
                fold_number = 0
                c = self.param_grid[param_ind][0]
                gamma = self.param_grid[param_ind][1]
                for train_index, test_index in cv:
#                    print("TRAIN:", train_index, "TEST:", test_index)
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = Y[train_index], Y[test_index]
                    gram = self.my_kernel(X_train, X_train, gamma)
                    my_svc = self.learner(C = c, kernel='precomputed', cache_size=2000)
                    my_svc.fit(gram, y_train)
                        
                    gram_test = self.my_kernel(X_train, X_test, gamma)

                    test_predictions = my_svc.predict(gram_test.T)
                    accs[fold_number] = self.my_loss(y_test, test_predictions)
                    fold_number += 1
                    
                self.scores[param_ind,i] = np.mean(accs)

            
#        min_ind = np.argmin(self.scores)
        min_ind = np.argmin(np.mean(self.scores, axis=1))

        with open(self.result_path, 'w') as res_file:
            print>>res_file, np.min(np.mean(self.scores, axis=1))
            print>>res_file, dict(C = self.param_grid[min_ind][0], Gamma = self.param_grid[min_ind][1])
            
            print>>res_file, np.std(self.scores, axis=1)
        
    def train_learner_cv(self, X, Y):
        """ """
        scaler = StandardScaler()

        X = scaler.fit_transform(X)
        
        for i in range(self.config.configuration["number_of_cvs"]):
            X_new, Y_new = shuffle(X, Y, random_state=i)
            cv = StratifiedKFold(y=Y_new, n_folds=self.config.configuration["number_of_cv_folds"])
            
            grid = GridSearchCV(self.learner(kernel = self.kernel,cache_size=2000), score_func = self.my_loss, param_grid=self.param_grid, 
                                cv=cv, n_jobs = 1)

            grid.fit(X_new, Y_new)
            
            for j in range(len(self.scores)):
                self.scores[j,i] = grid.grid_scores_[j][1]
            
            
        min_ind = np.argmin(np.mean(self.scores, axis=1))
        
        with open(self.result_path, 'w') as res_file:
            print>>res_file, np.min(np.mean(self.scores, axis=1))
            print>>res_file, grid.grid_scores_[min_ind][0]
            
            print>>res_file, np.std(self.scores, axis=1)
            
    def train_learner_opt(self, X, Y, X_test, Y_test, learner_params):
        """ """
#        scaler = StandardScaler()
#        X = scaler.fit_transform(X)
#        X_test = scaler.transform(X_test)
        X, Y, X_test, Y_test = self.scale_all_data(X, Y, X_test, Y_test)

        #TODO: add degree here
        if self.kernel == 'linear':
            clf = self.learner(kernel = self.kernel, cache_size=2000, C = learner_params['C'], probability = True)
        else:
            clf = self.learner(kernel = self.kernel, cache_size=2000, C = learner_params['C'], gamma = learner_params['gamma'], probability = True)
#        clf.fit(X, Y)  
##        Y_pred = clf.predict(X_test)
#        Y_pred_train = clf.predict(X)
##        error = self.my_loss(Y_test, Y_pred)
#        
#        Y_pred = clf.predict(X_test)
#        nonnan_indices = ~np.isnan(Y_test)
#        error = self.my_loss(Y_test[nonnan_indices], Y_pred[nonnan_indices])
#        self.logging.info('error is %s', str(error))
##        error = sum(Y_pred == Y_test) / float(len(Y_test))
#        with open(self.result_opt_path,'w') as res_file:
#            res_file.write(str(error))
#            res_file.write('\n')
#            res_file.write(' '.join(map(str,Y_pred) + ['\n']))
#            res_file.write(' '.join(map(str,Y_pred_train)))
#            res_file.write('\n')
#            res_file.write(str(learner_params))
        clf.fit(X, Y)

        self.fit_opt_learner(X, Y, X_test, Y_test, clf)
        

    def train_learner_opt_mykernel(self, X, Y, X_test, Y_test, learner_params):
        """ """
        
#        scaler = StandardScaler()
#        X = scaler.fit_transform(X)
#        X_test = scaler.transform(X_test)
        X, Y, X_test, Y_test = self.scale_all_data(X, Y, X_test, Y_test)

        c = learner_params['C']
        gamma = learner_params['Gamma']
        
        gram = self.my_kernel(X, X, gamma)
        my_svc = self.learner(C = c, kernel='precomputed', cache_size=2000)
        my_svc.fit(gram, Y)
                        
        gram_test = self.my_kernel(X, X_test, gamma)

        test_predictions = my_svc.predict(gram_test.T)
        error = self.my_loss(Y_test, test_predictions)
#        error = sum(test_predictions == Y_test)/float(len(Y_test))
        
        with open(self.result_opt_path,'w') as res_file:
            res_file.write(str(error))
            res_file.write('\n')
            res_file.write(' '.join(map(str,test_predictions) + ['\n']))
            res_file.write('\n')
            res_file.write(str(learner_params))
        
#    def set_output_file_path(self, res_path):
#        
#        self.result_path = res_path
#    
#    def set_output_file_opt_path(self, res_path):
#        
#        self.result_opt_path = res_path
    
    
    