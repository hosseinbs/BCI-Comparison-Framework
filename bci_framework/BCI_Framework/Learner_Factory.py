from sklearn.ensemble import RandomForestClassifier
import RandomForest_BCI
import SVM_Learner
import os
import sys
import Configuration_BCI
import numpy as np
import Data_Preprocessor as Data_Proc
import Error
import logging 
import GLM_Learner
import Boosting_learner
import GDA_Learner
# import MLP_Learner

class Learner_Factory:

    
    def __init__(self, config):
        """"""
        self.config = config


    def create_learner(self, learner_name, learner_params = None, random_seed = 0):
        
        if learner_name == 'MLP': 
            MLP_Learner = __import__('MLP_Learner')
            return MLP_Learner.MLP_Learner(self.config)
        
        elif learner_name == 'RANDOM_FOREST': 

            return RandomForest_BCI.RandomForest(self.config, method = 'classification')
            
    
        elif learner_name == 'RANDOM_FOREST_Reg': 

            return RandomForest_BCI.RandomForest(self.config, method = 'regression')
            
        elif learner_name == 'SVM':
            return SVM_Learner.SVM_Learner(self.config)
        
        elif learner_name == 'SVR':
            return SVM_Learner.SVM_Learner(self.config, method = 'regression')
        
#         elif learner_name == 'SVR_poly':
#             return SVM_Learner.SVM_Learner(self.config, 'poly' , method = 'regression')
# 
#         elif learner_name == 'SVR_rbf':
#             return SVM_Learner.SVM_Learner(self.config, 'rbf' , method = 'regression')
#         
#         elif learner_name == 'SVR_mykernel':
#             return SVM_Learner.SVM_Learner(self.config, 'mykernel' , method = 'regression')
        
        elif learner_name == 'linear':
            return GLM_Learner.GLM_Learner(self.config, method = 'regression')
            
        elif learner_name == 'LogisticRegression':
            return GLM_Learner.GLM_Learner(self.config, method = 'classification')
            
        elif learner_name == 'Boosting': 
            return Boosting_learner.BoostingLearner(self.config)
        
        elif learner_name == 'Boosting_Reg': 
            return Boosting_learner.BoostingLearner(self.config, 'regression')
        
        elif learner_name == 'LDA': 
            return GDA_Learner.GDA_Learner(self.config, type = 'LDA')
            
        elif learner_name == 'QDA':
            
            return GDA_Learner.GDA_Learner(self.config, type = 'QDA')
            
    
        