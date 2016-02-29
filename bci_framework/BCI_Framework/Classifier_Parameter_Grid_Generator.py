import numpy as np

class Classifier_Parameter_Grid_Generator:
    pass

    @staticmethod
    def generate_param_grid(config, learner_name, feature_param_list):
        
        if learner_name == 'RANDOM_FOREST': 

            RF_size = config.configuration["n_trees"]
            max_features_range = ["sqrt", "log2", None, 1]
            max_depth_range = [None, 15, 30, 50]
            min_samples_leaf_range = np.array([2, 5, 10])
            if feature_param_list is None:
                scores = np.zeros(shape=(len(max_features_range)*len(max_depth_range)* len(min_samples_leaf_range), config.configuration["number_of_cvs"], RF_size))
            
                param_grid = [ (m_feat, m_dep, m_sam_leaf) for m_feat in max_features_range for m_dep in max_depth_range 
                              for m_sam_leaf in min_samples_leaf_range]
            else:
                scores = np.zeros(shape=(len(max_features_range)*len(max_depth_range)* len(min_samples_leaf_range) * len(feature_param_list), config.configuration["number_of_cvs"], RF_size))
            
                param_grid = [ (m_feat, m_dep, m_sam_leaf, feat_param) for m_feat in max_features_range for m_dep in max_depth_range 
                              for m_sam_leaf in min_samples_leaf_range for feat_param in feature_param_list]
            
            
            return param_grid, scores
        
        elif learner_name == 'RANDOM_FOREST': 
            self.learner = RandomForest_BCI.RandomForest(config)
        