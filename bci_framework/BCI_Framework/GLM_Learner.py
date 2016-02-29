from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
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
from sklearn import linear_model

class GLM_Learner(NONRF_Learner):
    """applying Generalized Linear Machine to BCI dataset"""
    
    def __init__(self, config, method = 'classification'):
        """  """
        Learner.__init__(self, config, method)
        
#         self.regularization_type = regularization_type
#         
#         if method == 'classification' and regularization_type == 'l2':
#             
#             self.learner = linear_model.LogisticRegression
#             self.learnerCV = GridSearchCV(estimator=self.learner(), param_grid=dict(C=self.c_range), n_jobs=1, cv = self.config.configuration['number_of_cv_folds'])
#         
#         elif method == 'regression' and regularization_type == 'l2':
#         
#             self.learnerCV = linear_model.RidgeCV(cv = self.config.configuration['number_of_cv_folds'], max_iter=1000)
#             self.learner = linear_model.Ridge
#         
#         elif method == 'classification' and regularization_type == 'l1':
#             self.learner = linear_model.LogisticRegression
#             self.learnerCV = GridSearchCV(estimator=self.learner(), param_grid=dict(C=self.c_range), n_jobs=1, cv = self.config.configuration['number_of_cv_folds'])
#         
#         elif method == 'regression' and regularization_type == 'l1':
#             self.learnerCV = linear_model.LassoCV(cv = self.config.configuration['number_of_cv_folds'], max_iter=1000)
#             self.learner = linear_model.Lasso

#         self.logging.info('Done building a GLM_Learner instance: learner: %s learnerCV: %s ', self.learner.__class__, self.learnerCV.__class__)
    
    def generate_param_grid(self, feature_param_list, learner_name):
        
        regularization_type_range = ['l1', 'l2']
        c_range = np.logspace(-6, 1, 100)

        if feature_param_list is None:
            scores = np.zeros(shape=(len(c_range)*len(regularization_type_range), self.config.configuration["number_of_cvs_dict"][learner_name]))        
            
            param_grid = [ (reg_type, c) for reg_type in regularization_type_range for c in c_range]
            self.grid_dictionary = {'regularization_type':0, 'C':1}
        else:
            scores = np.zeros(shape=(len(c_range)*len(regularization_type_range)*len(feature_param_list), self.config.configuration["number_of_cvs_dict"][learner_name]))
            
            param_grid = [ (reg_type, c, feat_param) for reg_type in regularization_type_range for c in c_range
                          for feat_param in feature_param_list]
            self.grid_dictionary = {'regularization_type':0, 'C':1, 'fe_params':2}
        
        return param_grid, scores
    
    def set_params_list( self, learner_params, i):
        
        n_jobs = self.config.configuration["n_jobs"]

        reg_type = learner_params[0]
        c = learner_params[1]
        
        if self.method == 'classification':
            self.learner = linear_model.LogisticRegression(penalty = reg_type,  C = c)

        elif self.method == 'regression':
            if reg_type == 'l1':
                self.learner = linear_model.Lasso(alpha = c)
            elif reg_type == 'l2':
                self.learner = linear_model.Ridge(alpha = c)

    def set_params_dict( self, learner_params):
        
        n_jobs = self.config.configuration["n_jobs"]

        if self.method == 'classification':
            self.learner = linear_model.LogisticRegression(penalty = learner_params['regularization_type'],  C = learner_params['C'])

        elif self.method == 'regression':
            if learner_params['regularization_type'] == 'l1':
                self.learner = linear_model.Lasso(alpha = learner_params['C'])
            elif learner_params['regularization_type'] == 'l2':
                self.learner = linear_model.Ridge(alpha = learner_params['C'])

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
#                 print>>res_file, dict(regularization_type = param_grid[min_ind][0], C = param_grid[min_ind][1],
#                                     fe_params = None)
#             elif len(param_grid[0]) >= 3:
#                 print>>res_file, dict(regularization_type = param_grid[min_ind][0], C = param_grid[min_ind][1],
#                                    fe_params = param_grid[min_ind][2:])
#             
#             print>>res_file, np.std(scores, axis=1)
#             print>>res_file, scores   
#     
 
     
     
     
    
    
    
    
    
    
    
    
    
    
    
    
        
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

            self.learnerCV.fit(X_new, Y_new)
            
            if self.method == 'classification':
                for j in range(len(self.scores)):
                    self.scores[j,i] = 1 - self.learnerCV.grid_scores_[j][1]
            elif self.method == 'regression':
                self.scores[:, i] = np.mean(self.learnerCV.mse_path_, axis = 1)
            
        min_ind = np.argmin(np.mean(self.scores, axis=1))
        
        self.logging.info('Writing the results to file!')
        with open(self.result_path, 'w') as res_file:
            print>>res_file, np.min(np.mean(self.scores, axis=1))
            if self.method == 'regression':
                print>>res_file, {'alpha':self.learnerCV.alphas_[min_ind]}
            elif self.method == 'classification':
                print>>res_file, {'C':self.c_range[min_ind]}
            
            print>>res_file, np.std(self.scores, axis=1)            

    def train_learner_opt(self, X, Y, X_test, Y_test, learner_params):
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
            clf = self.learner(penalty = self.regularization_type,  C = learner_params['C'])
            self.logging.info('optimal logistic regression classifier trained with penalty type: %s and c = %s!', self.regularization_type, str(learner_params['C']))
        elif self.method == 'regression':
            clf = self.learner(alpha = learner_params['C'])
            self.logging.info('optimal linear regressor trained with alpha = %s!', str(learner_params['C']))
        
        clf.fit(X, Y)

        self.fit_opt_learner(X, Y, X_test, Y_test, clf)

#        clf.fit(X, Y)  
#        
#        Y_pred = clf.predict(X_test)
#        nonnan_indices = ~np.isnan(Y_test)
#        error = self.my_loss(Y_test[nonnan_indices], Y_pred[nonnan_indices])
#        self.logging.info('error is %s', str(error))
#
#        with open(self.result_opt_path,'w') as res_file:
#            res_file.write(str(error))
#            res_file.write('\n')
#            res_file.write(' '.join(map(str,Y_pred) + ['\n']))
#            res_file.write('\n')
#            res_file.write(str(learner_params))
        
