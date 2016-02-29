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
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn import cross_validation
from sklearn.metrics import classification_report
import json

class GDA_Learner(NONRF_Learner):
    """applying GDA to BCI dataset"""
    
    def __init__(self, config, type = 'LDA', method = 'classification'):
        """  """
        Learner.__init__(self, config, method)
        
        self.type = type
        

    def generate_param_grid(self, feature_param_list, learner_name):
        
        if feature_param_list is None:
            scores = np.zeros(shape=(1, self.config.configuration["number_of_cvs_dict"][learner_name]))        
            
            param_grid = [ None ]
            self.grid_dictionary = {}

        else:
            scores = np.zeros(shape=(len(feature_param_list), self.config.configuration["number_of_cvs_dict"][learner_name]))
            
            param_grid = [ (None, feat_param) for feat_param in feature_param_list]
            self.grid_dictionary = {'fe_params':1}
        
        return param_grid, scores
    
    def set_params_list( self, learner_params, i):
        
        n_jobs = self.config.configuration["n_jobs"]

        if self.type == 'LDA':
            self.learner = LDA()
        elif self.type == 'QDA':
            self.learner = QDA()

    def set_params_dict(self, learner_params):
        
        n_jobs = self.config.configuration["n_jobs"]

        if self.type == 'LDA':
            self.learner = LDA()
        elif self.type == 'QDA':
            self.learner = QDA()


#     def fit_calc_cv_scores(self, X_train, y_train, X_test, y_test):
#         
#         self.learner.fit(X_train, y_train)
#         return self.predict_error(X_test, y_test)
#     
#     def predict_error(self, X_test, Y_test):
# 
#         preds = self.learner.predict(X_test)
#         classification_error = np.sum((preds != Y_test))/float(len(Y_test))
#         precision, recall, _ , _ = precision_recall_fscore_support(Y_test, preds, average='weighted')
#         
#         return classification_error, precision, recall
    






        
    def train_learner(self,  X, Y, X_test = [], Y_test = [], learner_params = [] ,optimal = False):
        """  """
        if optimal:
            self.train_learner_opt(X, Y, X_test, Y_test, learner_params)
        else:
            self.train_learner_cv(X, Y)
    
    def train_learner_cv(self, Xs, Y, optimal = False):
        """ """
        self.logging.info('Standardizing data!')
        assert self.result_path != ''
#         X = np.asarray( X, dtype=np.float32, order='F')
#         Y = np.asarray( Y, dtype=np.short, order='F')
#         scaler = StandardScaler()

#         X = scaler.fit_transform(X)
        scaled_Xs = self.scale_training_data(Xs)
        
        self.logging.info('X size is: %s and Y size is: %s', '_'.join(map(str,scaled_Xs[0].shape)), map(str,Y.shape))
        
        for i in range(self.config.configuration["number_of_cvs"]):
            
            self.logging.info('iteration  number %s for cross validation', str(i))
            X_new, Y_new = shuffle(X, Y, random_state=i)
            
            scores = cross_validation.cross_val_score(self.learner, X_new, Y_new, cv=self.config.configuration["number_of_cv_folds"])
            
            if self.method == 'classification':
                self.scores[:,i] = 1 - np.mean(scores)
            elif self.method == 'regression':
                pass
#                self.scores[:, i] = np.mean(self.learnerCV.mse_path_, axis = 1)
            
#        clf = self.learner.fit(X,Y)
#        aa = clf.predict_proba(X)
        
        self.logging.info('Writing the results to file!')
        with open(self.result_path, 'w') as res_file:
            print>>res_file, np.mean(self.scores)
            print>>res_file,{}
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
            clf = self.learner
            self.logging.info('optimal GDA classifier trained')
        elif self.method == 'regression':
            pass
#            clf = self.learner()
#            self.logging.info('optimal linear regressor trained with alpha = %s!', str(learner_params['C']))
        
        clf.fit(X, Y)

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

        
