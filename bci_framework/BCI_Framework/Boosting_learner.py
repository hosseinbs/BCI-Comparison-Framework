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
from sklearn import ensemble
from sklearn.tree.tree import DecisionTreeClassifier, DecisionTreeRegressor


class BoostingLearner(NONRF_Learner):
    """applying Boosting algorithm to BCI dataset"""
    
    def __init__(self, config, method = 'classification'):
        """  """
        Learner.__init__(self, config, method)

#         self.RF_size = config.configuration["n_trees"]
#         self.n_jobs = config.configuration["n_jobs"]
#         
#         self.RF_size_range = [50, 100, 150, 200]
#         self.max_features_range = ["sqrt", "log2", None, 1]
#         self.max_depth_range = [1, 2, 3, 4, 5]
# #        self.min_density_range = np.array([0.1, 0.5, 1])
# #        self.loss_range = ['linear', 'exponential']
#         self.learning_rate_range = [0.1, 0.5, 1]
#         
#         self.scores = np.zeros(shape=(len(self.max_features_range)*len(self.max_depth_range) * 
#                                       len(self.learning_rate_range), self.config.configuration["number_of_cvs"]))
#         
#         self.param_grid = [ (m_rf_size, m_leran_rate, m_dep, m_feat,) for m_rf_size in self.RF_size_range
#                            for m_leran_rate in self.learning_rate_range for m_dep in self.max_depth_range for m_feat in self.max_features_range]
        
        
#         if method == 'classification':
#             self.learner = ensemble.AdaBoostClassifier
#             self.base_estimator = DecisionTreeClassifier
#         elif method == 'regression':
#             self.learner = ensemble.AdaBoostRegressor
#             self.base_estimator = DecisionTreeRegressor
# 
#         self.learnerCV = GridSearchCV(estimator=self.learner(), param_grid=dict(base_estimator__max_depth=self.max_depth_range, 
#                                                                               base_estimator__max_features=self.max_features_range, 
#                                                                               learning_rate = self.learning_rate_range, n_estimators = self.RF_size_range), n_jobs=self.n_jobs, 
#                                       cv = self.config.configuration['number_of_cv_folds'])
#         
#         
#         self.logging.info('A BoostingLearner Instance built with n_estimators: %s  number of CV folds: %s!' +
#                           'max_features_cv: %s max_depth_cv: %s learning_rate_cv: %s'+
#                           'learner: %s learnerCV: %s ',
#                           str(self.RF_size_range), str(self.config.configuration["number_of_cvs"]), str(self.max_features_range),
#                           str(self.max_depth_range), str(self.learning_rate_range),
#                           self.learner.__class__, self.learnerCV.__class__)

    def generate_param_grid(self, feature_param_list, learner_name):
        
        RF_size_range = [ 30, 40, 50, 70, 100, 150]
        max_features_range = ["sqrt", "log2", None, 1]
        max_depth_range = [1, 2, 3, 4, 5]
        learning_rate_range = [0.1, 0.5, 1]

        if feature_param_list is None:
            scores = np.zeros(shape=(len(max_features_range)*len(max_depth_range) * 
                                      len(learning_rate_range) * len(RF_size_range), 
                                      self.config.configuration["number_of_cvs_dict"][learner_name]))
        
            param_grid = [ (m_rf_size, m_leran_rate, m_dep, m_feat) for m_rf_size in RF_size_range
                           for m_leran_rate in learning_rate_range for m_dep in max_depth_range for m_feat in max_features_range]
            
            self.grid_dictionary = {'n_estimators':0, 'learning_rate':1, 'base_estimator__max_depth':2, 'base_estimator__max_features':3}

        else:
            scores = np.zeros(shape=(len(max_features_range)*len(max_depth_range) * len(RF_size_range) *
                                      len(learning_rate_range)* len(feature_param_list), self.config.configuration["number_of_cvs_dict"][learner_name]))
        
            param_grid = [ (m_rf_size, m_learn_rate, m_dep, m_feat, feat_param) for m_rf_size in RF_size_range
                           for m_learn_rate in learning_rate_range for m_dep in max_depth_range for m_feat in max_features_range
                           for feat_param in feature_param_list]
            self.grid_dictionary = {'n_estimators':0, 'learning_rate':1, 'base_estimator__max_depth':2, 'base_estimator__max_features':3, 'fe_params':4}

        return param_grid, scores
    
    def set_params_list(self, learner_params, i):
        
        m_rf_size = int(learner_params[0])
        m_learn_rate = learner_params[1]
        m_dep  = int(learner_params[2])
        m_feat  = learner_params[3]
        
        if self.method == 'classification':
            self.learner = ensemble.AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=m_dep, 
                                                               max_features=m_feat)
                            , n_estimators = int(m_rf_size), learning_rate=m_learn_rate)
        
        elif self.method == 'regression':
            self.learner = ensemble.AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=m_dep, 
                                                               max_features=m_feat)
                            , n_estimators = int(m_rf_size), learning_rate=m_learn_rate)
           
    def set_params_dict(self, learner_params):
        
        if self.method == 'classification':
            self.learner = ensemble.AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=learner_params['base_estimator__max_depth'], 
                                                               max_features=learner_params['base_estimator__max_features'])
                            , n_estimators = int(learner_params['n_estimators']), learning_rate=learner_params['learning_rate'])
        
        elif self.method == 'regression':
            self.learner = ensemble.AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=learner_params['base_estimator__max_depth'], 
                                                               max_features=learner_params['base_estimator__max_features'])
                            , n_estimators = int(learner_params['n_estimators']), learning_rate=learner_params['learning_rate'])
     
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
#             if len(param_grid[0]) == 4:
#                 print>>res_file, dict(n_estimators = param_grid[min_ind][0], learning_rate = param_grid[min_ind][1],
#                                    base_estimator__max_depth = param_grid[min_ind][2], base_estimator__max_features =  param_grid[min_ind][3], fe_params = None)
#             elif len(param_grid[0]) >= 5:
#                 print>>res_file, dict(n_estimators = param_grid[min_ind][0], learning_rate = param_grid[min_ind][1],
#                                    base_estimator__max_depth = param_grid[min_ind][2], base_estimator__max_features =  param_grid[min_ind][3], fe_params = param_grid[min_ind][4:])
#             
#             print>>res_file, np.std(scores[min_ind,:])
#             print>>res_file, scores   









          
          
            
    def train_learner_cv(self, X, Y, optimal = False):
        """  """
        assert self.result_path != ''
        self.logging.info('Standardizing data!')
        X = np.asarray( X, dtype=np.float32, order='F')
        Y = np.asarray( Y, dtype=np.short, order='F')
        scaler = StandardScaler()

        X = scaler.fit_transform(X)
        
        self.logging.info('X size is: %s and Y size is: %s', '_'.join(map(str,X.shape)), map(str,Y.shape))

        for i in range(self.config.configuration["number_of_cvs"]):
            
            self.logging.info('iteration  number %s for cross validation', str(i))
            X_new, Y_new = shuffle(X, Y, random_state=i)

            self.learnerCV.fit(X_new, Y_new)
            
#           TODO: check error func!!!!!!!!!!!!! 
            for j in range(len(self.scores)):
                if self.method == 'classification':
                    self.scores[j,i] = 1 - self.learnerCV.grid_scores_[j][1]
                elif self.method == 'regression':
                    self.scores[j,i] = self.learnerCV.grid_scores_[j][1]
            
        min_ind = np.argmin(np.mean(self.scores, axis=1))
        
        self.logging.info('Writing the results to file!')
        with open(self.result_path, 'w') as res_file:
            print>>res_file, np.min(np.mean(self.scores, axis=1))
            
            print>>res_file, self.learnerCV.grid_scores_[min_ind][0]
                       
            print>>res_file, np.std(self.scores, axis=1)
            
    def train_learner_opt(self, X, Y, X_test, Y_test, learner_params):
        """ """
        
        X, Y, X_test, Y_test = self.scale_all_data(X, Y, X_test, Y_test)

        clf =  self.learner(base_estimator=self.base_estimator(max_depth=learner_params['base_estimator__max_depth'], 
                                                               max_features=learner_params['base_estimator__max_features'])
                            , n_estimators = int(learner_params['n_estimators']), learning_rate=learner_params['learning_rate'])
    
        clf.fit(X, Y)
        self.fit_opt_learner(X, Y, X_test, Y_test, clf)
