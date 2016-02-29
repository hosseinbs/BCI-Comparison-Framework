from sklearn.metrics import zero_one_loss, mean_squared_error
import logging
from sklearn.preprocessing import StandardScaler
import numpy as np
import cPickle
import Learner_Factory
import Classifier_Parameter_Grid_Generator
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedKFold, cross_val_score
import json

class Learner_Manager:

    def __init__(self, config, learner_name, feature_name):
        """   """
        self.feature_name = feature_name
        self.learner_name = learner_name
        self.logging = logging
        if config.configuration['logging_level_str'] == 'INFO':
            self.logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
        else:
            self.logging.basicConfig(level=logging.NOTSET)
        
        self.logging.info('started building a learner!') 
        self.config = config
        self.my_learner_factory = Learner_Factory.Learner_Factory(config)
        
    def train_learner(self,  Xs, Y, X_test = [], Y_test = [], learner_params = [] ,optimal = False):
        """ """
        if optimal:
            self.fit_opt_learner(Xs, Y, X_test, Y_test, learner_params)
        else:
            self.train_learner_cv(Xs, Y)
    
    def scale_training_data(self, Xs):
        
        scaled_Xs = []
        for X in Xs:
            X = np.asarray( X, dtype=np.float32, order='F')
            scaler = StandardScaler()

            X = scaler.fit_transform(X)
            
            scaled_Xs.append(X)
        
        return scaled_Xs
        
    def scale_all_data(self, X, Y, X_test, Y_test):
        ''' '''
        self.logging.info('Standardizing data!')
        Y_test = np.array(Y_test)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_test = scaler.transform(X_test)
        self.logging.info('X size is: %s and Y size is: %s and X_test size is: %s and Y_test size is: %s',
                           '_'.join(map(str,X.shape)), str(len(Y)), '_'.join(map(str,X_test.shape)), str(len(Y_test)))
        
        return X, Y, X_test, Y_test

    def fit_opt_learner(self, X, Y, X_test, Y_test, learner_params):
        ''' '''
        
#         clf.fit(X, Y)
        X, Y, X_test, Y_test = self.scale_all_data(X[0], Y, X_test[0], Y_test)
        learner = self.my_learner_factory.create_learner(self.learner_name)
        learner.set_params_dict(learner_params)
        learner.fit(X, Y)
        Y_pred_train = learner.predict(X)
        
        Y_pred = learner.predict(X_test)
        nonnan_indices = ~np.isnan(Y_test)
        error = learner.test_phase_loss(Y_test[nonnan_indices], Y_pred[nonnan_indices])
        if type(error) == list:
            for err in error:
                self.logging.info('error is %s\n', str(err))
        else:
            self.logging.info('error is %s', str(error))

        probs_train = learner.learner.predict_proba(X)
        probs_test = learner.learner.predict_proba(X_test)

        self.logging.info('Writing final results to file ')

        np.savez(self.result_opt_path, error=error, Y_pred=Y_pred, Y_pred_train=Y_pred_train, probs_train=probs_train, probs_test = probs_test)
        
        self.logging.info('Writing optimal classifier to file ')
        
        # save the classifier
#         with open(self.result_opt_classifier_path + '.pkl', 'wb') as fid:
#             cPickle.dump(learner.learner, fid)
            
    @staticmethod
    def find_cv_error(res_file_name):
        """ """
        params = {}
        final_results_dict = json.load(open(res_file_name))
        error = final_results_dict['error']
        params = final_results_dict['error_params']
        return error, params
        
#         with open(res_file_name, 'r') as res_file:
#             error = float(res_file.readline())
#             params = eval(res_file.readline())
#             return error, params
    
    @staticmethod
    def find_opt_error(opt_res_file_name):
        """ """
#        params = {}
        with open(opt_res_file_name, 'r') as res_file:
            error = float(res_file.readline())
#            params = eval(res_file.readline())
            return error

    def train_learner_cv(self, Xs, Y, optimal = False):
        
        
        assert self.result_path != ''
        my_learner = self.my_learner_factory.create_learner(self.learner_name)
        
        Y = np.asarray( Y, dtype=np.short, order='F')
        scaled_Xs = self.scale_training_data(Xs)
        if self.feature_name in (self.config.configuration['fe_params_dict']).keys():
            
            feature_param_values = (self.config.configuration['fe_params_dict'])[self.feature_name]
            feature_param_val_indices = dict(zip( feature_param_values, range(len(feature_param_values))))
            param_grid, scores = my_learner.generate_param_grid(feature_param_values, self.learner_name)
        else:
            feature_param_values = None
            param_grid, scores = my_learner.generate_param_grid(None, self.learner_name)
        
        precision_scores, recall_scores = np.zeros(shape = scores.shape), np.zeros(shape = scores.shape)
        
        for i in range(self.config.configuration["number_of_cvs_dict"][self.learner_name]):
            
            for param_ind in range(len(scores)):
                print param_grid[param_ind]
                
                my_learner.set_params_list(param_grid[param_ind], i)
                
                if feature_param_values is None:
                    X = scaled_Xs[0]
                else:
                    X = scaled_Xs[feature_param_val_indices[param_grid[param_ind][-1]]]
                X_new, Y_new = shuffle(X, Y, random_state = i)
                
                cv = StratifiedKFold(y = Y_new, n_folds = self.config.configuration["number_of_cv_folds"])
                
                if len(scores.shape) == 3:
                    #this is for ensemble learning methods # Only RF
                    cv_errors_sum = np.zeros(scores.shape[2])
                    precision_sum = np.zeros(scores.shape[2])
                    recall_sum = np.zeros(scores.shape[2])
                elif len(scores.shape) == 2:
                    cv_errors_sum, precision_sum, recall_sum = 0, 0, 0
                
                for train_index, test_index in cv:
#                    print("TRAIN:", train_index, "TEST:", test_index)
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = Y[train_index], Y[test_index]
                    cv_error_temp, precision_temp, recall_temp = my_learner.fit_calc_cv_scores(X_train, y_train, X_test, y_test)
                    cv_errors_sum += cv_error_temp
                    precision_sum += precision_temp
                    recall_sum += recall_temp
#                     my_learner.fit(X_train, y_train)

#                     test_predictions = my_learner.predict(X_test)
#                     cv_errors_sum += self.predict_forall_estimators(X_test, y_test)
                
                crossval_error = cv_errors_sum/self.config.configuration["number_of_cv_folds"]
                precision = precision_sum/self.config.configuration["number_of_cv_folds"]
                recall = recall_sum/self.config.configuration["number_of_cv_folds"]
                print 'error = ', crossval_error
                
                if len(scores.shape) == 3:
                    scores[param_ind, i, :] = crossval_error
                    precision_scores[param_ind, i, :] = precision
                    recall_scores[param_ind, i, :] = recall
                else:
                    scores[param_ind, i] = crossval_error
                    precision_scores[param_ind, i] = precision
                    recall_scores[param_ind, i] = recall
                
        
        my_learner.write_cv_results_toFile(scores, precision_scores, recall_scores, param_grid, self.result_path)
                 
    def set_output_file_path(self, res_path):
        
        self.result_path = res_path
        
    def set_output_file_opt_path(self, res_path):
        
        self.result_opt_path = res_path
    
    def set_output_file_opt_classifier(self, res_path):
        self.result_opt_classifier_path = res_path
    