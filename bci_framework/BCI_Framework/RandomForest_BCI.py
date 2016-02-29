from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
import numpy as np
import pickle
import sys
from Learner import Learner
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import pairwise, zero_one_loss, mean_squared_error
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.utils import shuffle
from scipy.stats import mode
import json
from sklearn.metrics import precision_recall_fscore_support

class RandomForest(Learner):
    """applying random forest to BCI dataset"""
    
    def __init__(self, config, method='classification'):
        """  """
        Learner.__init__(self, config, method)
    
    def generate_param_grid(self, feature_param_list, learner_name):
        
        RF_size = self.config.configuration["n_trees"]
        max_features_range = ["sqrt", "log2", None, 1]
        max_depth_range = [None, 15, 30, 50]
        min_samples_leaf_range = np.array([2, 5, 10])
        
        if feature_param_list is None:
            scores = np.zeros(shape=(len(max_features_range)*len(max_depth_range)*
                                      len(min_samples_leaf_range), self.config.configuration["number_of_cvs_dict"][learner_name], RF_size))
        
            param_grid = [ (m_feat, m_dep, m_sam_leaf) for m_feat in max_features_range for m_dep in max_depth_range 
                          for m_sam_leaf in min_samples_leaf_range]
            self.grid_dictionary = {'max_features':0, 'max_depth':1, 'min_samples_leaf':2}

        else:
            scores = np.zeros(shape=(len(max_features_range)*len(max_depth_range)* len(min_samples_leaf_range) * len(feature_param_list), 
                                     self.config.configuration["number_of_cvs_dict"][learner_name], RF_size))
        
            param_grid = [ (m_feat, m_dep, m_sam_leaf, feat_param) for m_feat in max_features_range for m_dep in max_depth_range 
                          for m_sam_leaf in min_samples_leaf_range for feat_param in feature_param_list]
            self.grid_dictionary = {'max_features':0, 'max_depth':1, 'min_samples_leaf':2, 'fe_params':3}

        
        return param_grid, scores
    
    def set_params_list( self, learner_params, i):
        
        RF_size = self.config.configuration["n_trees"]
        n_jobs = self.config.configuration["n_jobs"]

        m_feat = learner_params[0]
        m_dep = learner_params[1]
        m_sam_leaf  = learner_params[2]
        
        if self.method == 'classification':
            self.learner = RandomForestClassifier(n_estimators = RF_size, 
                                                          oob_score = True, n_jobs= n_jobs,
                                                          max_depth = m_dep, max_features = m_feat,
                                                          min_samples_leaf = m_sam_leaf, random_state= i)

        elif self.method == 'regression':
            self.learner = RandomForestRegressor(n_estimators = RF_size, 
                                                         oob_score = True, n_jobs= n_jobs,
                                                         max_depth = m_dep, max_features = m_feat,
                                                         min_samples_leaf = m_sam_leaf, random_state= i)

    def set_params_dict( self, learner_params):
        
        if 'n_trees' in learner_params.keys():
            RF_size = int(learner_params["n_trees"])
        else:
            RF_size = self.config.configuration["n_trees"]
        
        n_jobs = self.config.configuration["n_jobs"]

        m_feat = learner_params['max_features']
        m_dep = learner_params['max_depth']
        m_sam_leaf  = learner_params['min_samples_leaf']
        
        if self.method == 'classification':
            self.learner = RandomForestClassifier(n_estimators = RF_size, 
                                                          oob_score = True, n_jobs= n_jobs,
                                                          max_depth = m_dep, max_features = m_feat,
                                                          min_samples_leaf = m_sam_leaf, random_state= 0)

        elif self.method == 'regression':
            self.learner = RandomForestRegressor(n_estimators = RF_size, 
                                                         oob_score = True, n_jobs= n_jobs,
                                                         max_depth = m_dep, max_features = m_feat,
                                                         min_samples_leaf = m_sam_leaf, random_state= 0)

    def fit_calc_cv_scores(self, X_train, y_train, X_test, y_test):
        
        self.learner.fit(X_train, y_train)
        return self.predict_forall_estimators(X_test, y_test)
    
    def predict_forall_estimators(self, X_test, Y_test):

        n_estimators = len(self.learner.estimators_) 
        errors, precisions, recalls = np.zeros(shape = (n_estimators)), np.zeros(shape = (n_estimators)), np.zeros(shape = (n_estimators))
        errors[:] = np.NaN
         
        predictions = np.empty(shape = (X_test.shape[0], n_estimators))
        predictions[:] = np.NAN
        
        for i in range(0,n_estimators):
            predictions[:,i] = self.learner.estimators_[i].predict(X_test) + 1
            
            most_common = mode(predictions[:,0:i+1], axis = 1)
            preds_uptonow = [item for sublist in most_common[0] for item in sublist]
            errors[i] = np.sum((preds_uptonow != Y_test))/float(len(Y_test))
            precisions[i], recalls[i], _ , _ = precision_recall_fscore_support(Y_test, preds_uptonow, average='weighted')

#         print predictions
        return errors, precisions, recalls
    
    def write_cv_results_toFile(self, scores, precision_scores, recall_scores, param_grid, result_path):
        
        final_results_dict = {}
        all_scores_dict = {'error' : scores, 'precision' : precision_scores, 'recall' : recall_scores}
        for score in all_scores_dict:
            avg_score = np.mean(all_scores_dict[score], axis=1)
            std_score = np.std(all_scores_dict[score], axis=1)

            if score == 'error':
                opt_ind = np.unravel_index(np.argmin(avg_score), avg_score.shape)
            else:
                opt_ind = np.unravel_index(np.argmax(avg_score), avg_score.shape)
            
            final_results_dict[score] = avg_score[opt_ind]
            final_results_dict[score + '_std'] = std_score[opt_ind]
            params_dict = {}
            for element in self.grid_dictionary:
                params_dict[element] =  param_grid[opt_ind[0]][self.grid_dictionary[element]]
            final_results_dict[score + '_params'] = params_dict
            n_estimators = opt_ind[1] + 1
            final_results_dict[score + '_params'].update(dict(n_trees = str(n_estimators)))
            if not 'fe_params' in params_dict:
                final_results_dict[score + '_params'].update(dict(fe_params = None))
        
        final_results_dict['channel_type'] = self.config.channel_type

        json.dump(final_results_dict, open(result_path,'w'))  

    
    
    
    
    
    
    
    
    
    
    
    
    
    def train_learner(self,  X, Y, X_test = [], Y_test = [], learner_params = [] ,optimal = False):
        """  """
        if optimal:
            self.train_learner_opt(X, Y, X_test, Y_test, learner_params)
        else:
            self.train_learner_cv(X, Y)
    
    def train_learner_cv(self, X, Y, optimal = False):
        
        assert self.result_path != ''
        X = np.asarray( X, dtype=np.float32, order='F')
        Y = np.asarray( Y, dtype=np.short, order='F')
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        for i in range(self.config.configuration["number_of_cvs"]):
            
            for param_ind in range(len(self.scores)):
                print self.param_grid[param_ind]

                m_feat = self.param_grid[param_ind][0]
                m_dep = self.param_grid[param_ind][1]
#                m_dens = self.param_grid[param_ind][2]
                m_sam_leaf  = self.param_grid[param_ind][2]
                
                if self.method == 'classification':
                    self.learner = RandomForestClassifier(n_estimators = self.RF_size, 
                                                          oob_score=True, n_jobs= self.n_jobs,
                                                          max_depth = m_dep, max_features = m_feat,
                                                          min_samples_leaf = m_sam_leaf, random_state=i)
                elif self.method == 'regression':
                    self.learner = RandomForestRegressor(n_estimators = self.RF_size, 
                                                         oob_score=True, n_jobs= self.n_jobs,
                                                         max_depth = m_dep, max_features = m_feat,
                                                         min_samples_leaf = m_sam_leaf, random_state=i)
# TODO: check regression error                
#                self.learner.fit(X, Y)
#                oob_err = self.learner.oob_score_
#                crossval_error = 1 - cross_val_score(self.learner, X, Y,  cv=5).mean()
                X_new, Y_new = shuffle(X, Y, random_state=i)
                cv = StratifiedKFold(y=Y_new, n_folds=self.config.configuration["number_of_cv_folds"])
                cv_errors_sum = np.zeros(self.RF_size)
                for train_index, test_index in cv:
#                    print("TRAIN:", train_index, "TEST:", test_index)
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = Y[train_index], Y[test_index]
                    self.learner.fit(X_train, y_train)

                    test_predictions = self.learner.predict(X_test)
                    cv_errors_sum += self.predict_forall_estimators(X_test, y_test)
                
                crossval_error = cv_errors_sum/self.config.configuration["number_of_cv_folds"]
                print 'error = ', crossval_error
                
#                bb = self.__calc_OOB_inbag_scores( X, Y, self.learner)    
#                self.scores[param_ind, i] = oob_err
                self.scores[param_ind, i, :] = crossval_error
                
        
        avg_errs = np.mean(self.scores, axis=1)
        min_ind = np.unravel_index(np.argmin(avg_errs), avg_errs.shape)
        
        with open(self.result_path, 'w') as res_file:
            print>>res_file, np.min(np.mean(self.scores, axis=1))
            print>>res_file, dict(max_features = self.param_grid[min_ind[0]][0], max_depth = self.param_grid[min_ind[0]][1],
                                   min_samples_leaf = self.param_grid[min_ind[0]][2], n_trees = min_ind[1])
            
            print>>res_file, np.std(self.scores, axis=1)
            print>>res_file, self.scores            

    def train_learner_opt(self, X, Y, X_test, Y_test, learner_params):
        """ """
        
#        Y_test = np.array(Y_test)
#        self.logging.info('Standardizing data!')
#        scaler = StandardScaler()
#        X = scaler.fit_transform(X)
#        X_test = scaler.transform(X_test)
#        self.logging.info('X size is: %s and Y size is: %s and X_test size is: %s and Y_test size is: %s',
#                           '_'.join(map(str,X.shape)), str(len(Y)), '_'.join(map(str,X_test.shape)), str(len(Y_test)))
        
        X, Y, X_test, Y_test = self.scale_all_data(X, Y, X_test, Y_test)

        clf = self.learner_opt(n_estimators = learner_params["n_trees"], oob_score=True, n_jobs= self.n_jobs, 
                                             compute_importances = True, max_features = learner_params["max_features"],
                                             max_depth = learner_params["max_depth"], min_samples_leaf = learner_params["min_samples_leaf"])
    
        clf.fit(X, Y)

        self.fit_opt_learner(X, Y, X_test, Y_test, clf)

#        clf.fit(X, Y)  
#        Y_pred_train = clf.predict(X)
#        self.logging.info('optimal Random Forest trained with penalty type: %s and parameters = %s!', self.my_loss.__class__, learner_params)
#        self.logging.info('calculating inbag and oob scores!')
#        
##        oob_predictions, inbag_predictions, oob_scores, inbag_scores = self.__calc_OOB_inbag_scores(X, Y, clf)
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
#            res_file.write(' '.join(map(str,Y_pred_train)))
#            res_file.write('\n')
#            res_file.write(str(learner_params))
#            res_file.write('\n')
#            res_file.write(str(oob_scores))
#            res_file.write('\n')
#            res_file.write(str(oob_scores))
#            res_file.write('\n')

    def __calc_OOB_inbag_scores(self, X, Y, clf):
        
        class_labels = list(set(Y))
        num_of_classes = len(class_labels)
        oob_predictions = np.zeros([num_of_classes, len(Y)])
        inbag_predictions = np.zeros([num_of_classes, len(Y)])
        
        oob_scores = np.zeros(self.RF_size)
        inbag_scores = np.zeros(self.RF_size)
        
        indices = np.array(range(len(Y)))
    
        for index in range(self.RF_size):
            
            d3 = clf.estimators_[index]
            current_tree_predictions = d3.predict(X)
            
            oob_predictions = self.__calc_current_tree_prediction(oob_predictions, current_tree_predictions, 
                                                                  ~d3.indices_, indices, class_labels) ## OOB scores
            inbag_predictions = self.__calc_current_tree_prediction(inbag_predictions, current_tree_predictions, 
                                                                    d3.indices_, indices, class_labels) ## inbag scores
        
            oob_scores[index] = self.__calc_scores(oob_predictions,Y) ## calc oob scores
            inbag_scores[index] = self.__calc_scores(inbag_predictions,Y) ## calc oob scores
        
                
        return oob_predictions, inbag_predictions, oob_scores, inbag_scores
    
    def __calc_scores(self, given_predictions, Y):
        
        class_labels = list(set(Y))
        all_predictions = np.argmax(given_predictions,0)
        if len(class_labels) == 3:
            all_predictions -= 1 # TODO: why is it like this?
        elif len(class_labels) > 3:
            sys.exit('Error! __calc_scores forest')
          
        number_of_matches = (all_predictions == Y).tolist().count(True)
        return float(number_of_matches)/len(Y)  
    
    def __calc_current_tree_prediction(self, sample_predictions, current_tree_predictions, given_indices, all_indices, class_labels):
        
        current_samples_predictions = current_tree_predictions[given_indices]
            
        current_samples_indices = all_indices[given_indices]
        
        for index in range(len(class_labels)):
            label = class_labels[index]
            sample_predictions[index,current_samples_indices[current_samples_predictions == label]] += 1
        
        return sample_predictions
