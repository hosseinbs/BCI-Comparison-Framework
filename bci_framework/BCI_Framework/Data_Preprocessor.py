import Configuration_BCI
import numpy as np
import os
import Feature_Extractor as FE
import Filter
import sys
from spatfilt import *
from sklearn.preprocessing import StandardScaler
import itertools
import logging

class Data_Preprocessor:
    """ """
    
    def __init__(self, config, subject, feature_extractor_name, number_of_CSPs):
        """"""
        self.logging = logging
        if config.configuration['logging_level_str'] == 'INFO':
            self.logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
        else:
            self.logging.basicConfig(level=logging.NOTSET)
        
        self.logging.info('begin creating Data_Preprocessor')

        self.number_of_CSPs = number_of_CSPs
        self.config = config
        self.problem_type = self.config.configuration['dataset_type_str']

        self.set_feature_extractor(feature_extractor_name)
        self.set_filter()
        self.logging.info('A new Data_Preprocessor is created: number_of_CSPs: %s  BCI type: %s feature_extractor_name: %s', str(self.number_of_CSPs), self.problem_type, feature_extractor_name)
            
    def set_feature_extractor(self, feature_extractor_name):
        
        self.feature_extractor_name = feature_extractor_name
        FE.Feature_Extractor.config = self.config
        if feature_extractor_name == 'BP':
            self.features_path = os.path.join( self.config.configuration['train_data_dir_name_str'], 'BP')
            self.feature_extractor = FE.Feature_Extractor.bp_feature_extractor

#         elif feature_extractor_name == 'CSP':
#             self.features_path = os.path.join( self.config.configuration['train_data_dir_name_str'], 'CSP')
#             self.feature_extractor = FE.Feature_Extractor.bp_feature_extractor

        elif feature_extractor_name == 'raw':
            self.features_path = os.path.join( self.config.configuration['train_data_dir_name_str'], 'RAW')
            self.feature_extractor = FE.Feature_Extractor.raw_feature_extractor
        
        elif feature_extractor_name == 'wackerman':
            self.features_path = os.path.join( self.config.configuration['train_data_dir_name_str'], 'wackerman')
            self.feature_extractor = FE.Feature_Extractor.wackerman_feature_extractor
        
        elif feature_extractor_name == 'logbp':
            self.features_path = os.path.join( self.config.configuration['train_data_dir_name_str'], 'logbp')
            self.feature_extractor = FE.Feature_Extractor.logbp_feature_extractor
        
        elif feature_extractor_name == 'morlet':
        
            self.features_path = os.path.join( self.config.configuration['train_data_dir_name_str'], 'morlet')
            self.feature_extractor = FE.Feature_Extractor.morlet_feature_extractor
        
        elif feature_extractor_name == 'AR':
        
            self.features_path = os.path.join( self.config.configuration['train_data_dir_name_str'], 'AR')
            self.feature_extractor = FE.Feature_Extractor.ar_feature_extractor
            
        else:
            sys.stderr.write("error: the feature extraction method does not exist\n" )
            sys.exit(1)

    def set_filter(self):

        self.apply_filter = Filter.Filter.butter_bandpass_filter
#        self.config.configuration['filter_name_str'] = self.config.configuration['filter_name_str'].replace("\n", "")
#        print self.config.configuration['filter_name_str']
#        if self.config.configuration['filter_name_str']== 'butter':
#            self.apply_filter = Filter.Filter.butter_bandpass_filter
#        elif self.config.configuration['filter_name_str']== 'chebyshev':###################### TODO: add chebyshev filter
#            self.apply_filter = Filter.Filter.butter_bandpass_filter
#        else:
#            sys.stderr.write("error: Filtering method does not exist\n" )
#            sys.exit(1)

    def load_dataset_train(self,subject):
        """Load the data for each subject"""
        
#        print(os.path.realpath('.'))
        train_data_folder = self.config.configuration['train_data_dir_name_str']
#        print(os.getcwd())
        X = np.loadtxt(os.path.join(train_data_folder, subject + '_X.txt'))
        Y = np.loadtxt(os.path.join(train_data_folder, subject + '_Y.txt'))
        
        which_channels = self.config.which_channels
        X = X[:,np.array(which_channels)]        
                
        return X, Y

    def load_dataset_test(self,subject):
        """Load the data for each subject"""
        test_data_folder = self.config.configuration['test_data_dir_name_str']
        
        X_test = np.loadtxt(os.path.join(test_data_folder, subject + '_X.txt'), dtype=np.float32)
        Y_test = np.loadtxt(os.path.join(test_data_folder, subject + '_Y.txt'))

        which_channels = self.config.which_channels
        X_test = X_test[:,np.array(which_channels)]
        
        return X_test, Y_test

    def extract_data_samples_forall_feature_params(self, subject, params_dict, optimal, cutoff_frequencies_low_list = None, cutoff_frequencies_high_list = None):

        Xs = []
        if params_dict['fe_params'] == None:
                
            X_temp, Y = self.extract_data_samples(subject, params_dict, optimal, params_dict['fe_params'], 
                                                  cutoff_frequencies_low_list, cutoff_frequencies_high_list)
            Xs.append(X_temp)
        elif isinstance(params_dict['fe_params'], (int, long, float)):
            X_temp, Y = self.extract_data_samples(subject, params_dict, optimal, params_dict['fe_params'], 
                                                  cutoff_frequencies_low_list, cutoff_frequencies_high_list)
            Xs.append(X_temp)
            
        else:
            for ex_param in params_dict['fe_params']:
                X_temp, Y = self.extract_data_samples(subject, params_dict, optimal, ex_param, cutoff_frequencies_low_list, cutoff_frequencies_high_list)
                Xs.append(X_temp)
            

        return Xs, Y
    
    
    def extract_data_samples(self, subject, params_dict, optimal, fe_param, cutoff_frequencies_low_list = None, cutoff_frequencies_high_list = None):
        """ """
        
        subject_number = self.config.configuration["subject_names_str"].index(subject)
        
        if optimal:
            raw_X, raw_Y = self.load_dataset_test(subject)###############################################################
            self.logging.info("started extracting features from testing data for subject: %s", subject)
        else:
            raw_X, raw_Y = self.load_dataset_train(subject)
            self.logging.info("started extracting features from training data for subject: %s", subject)

        if cutoff_frequencies_low_list is None and self.feature_extractor_name != 'morlet':
            cutoff_frequencies_low = np.tile(np.array(self.config.configuration["cutoff_frequencies_low_list"]).T,(raw_X.shape[1],1))
            cutoff_frequencies_high = np.tile(np.array(self.config.configuration["cutoff_frequencies_high_list"]).T,(raw_X.shape[1],1))
        elif cutoff_frequencies_low_list is None and self.feature_extractor_name == 'morlet': # this part is too messy fix this later
            cutoff_frequencies_low = np.tile(np.array([0.5]).T,(raw_X.shape[1],1))
            cutoff_frequencies_high = np.tile(np.array([30]).T,(raw_X.shape[1],1))
        else :
            cutoff_frequencies_low_list = cutoff_frequencies_low_list.split('_')
            cutoff_frequencies_high_list = cutoff_frequencies_high_list.split('_')
            cutoff_frequencies_low_list = np.array([float(e) for e in cutoff_frequencies_low_list])
            cutoff_frequencies_high_list = np.array([float(e) for e in cutoff_frequencies_high_list])
            cutoff_frequencies_low = np.reshape(cutoff_frequencies_low_list, (raw_X.shape[1],-1))
            cutoff_frequencies_high = np.reshape(cutoff_frequencies_high_list, (raw_X.shape[1],-1))
#            cutoff_frequencies_low = np.tile(np.array(cutoff_frequencies_low_list.T,(raw_X.shape[1],1)))
#            cutoff_frequencies_high = np.tile(np.array(cutoff_frequencies_high_list.T,(raw_X.shape[1],1)))
        
        self.logging.info("number of filters is: %s", str(cutoff_frequencies_low.shape[1]))
        for filt_number in range(cutoff_frequencies_low.shape[1]):

            
            cutoff_freq_low = cutoff_frequencies_low[:,filt_number]
            cutoff_freq_high = cutoff_frequencies_high[:,filt_number]
            
            self.logging.info("for filter number %s low frequencies are %s and high frequencies are %s",
                               str(filt_number), str(cutoff_freq_low), str(cutoff_freq_high))
            
            raw_X = np.array(raw_X)
            filtered_X = np.copy(raw_X) ##copying the list- always copy lists
            self.logging.info("raw_X.shape is: %s", str(raw_X.shape))
            for i in range(raw_X.shape[1]):
                filtered_X[:,i] = self.apply_filter(filtered_X[:,i], cutoff_freq_low[i], cutoff_freq_high[i], self.config.configuration["sampling_rate"])
            
            if self.number_of_CSPs != -1:
                filtered_X = self.apply_CSP(filtered_X, raw_Y)
                self.logging.info('After applying CSP filters X size is: %s', str(filtered_X.shape))
            
            if self.config.configuration["dataset_type_str"] == 'async':
                
                if optimal:
                    self.logging.info('started extracting features from the test data for async BCI!')
                    X_new, Y_new = self.prepare_samples_async_opt(subject_number, filtered_X, raw_Y, params_dict, fe_param)
                else:
                    self.logging.info('started extracting features from the training data for async BCI!')
                    X_new, Y_new = self.prepare_samples_async(subject_number, filtered_X, raw_Y, params_dict, fe_param)
                    
            elif self.config.configuration["dataset_type_str"] == 'sync':
                self.logging.info('started extracting features from data for sync BCI!')
                X_new, Y_new = self.prepare_samples_sync(subject_number, filtered_X, raw_Y, params_dict, fe_param)
                
            if filt_number == 0:
                X = np.array(X_new)
            else:
                X = np.concatenate((X, np.array(X_new)), axis=1)
            
            self.logging.info('After applying filter %s size of X is %s', str(filt_number), str(X.shape))
        return X, Y_new
    
    def prepare_samples_async_opt(self, subject_number, X_test_raw, Y_test_raw, params_dict, fe_params):    
        """  """
        
        subject = self.config.configuration['subject_names_str'][subject_number]
        window_size = int(params_dict['window_size'])
        window_overlap_size = 1
        
#        X_test_raw, Y_test_raw = self.load_dataset_test(subject)
        x_shape = [0,0]
        x_shape[0] = len(range(0, len(X_test_raw) - window_size + 1, window_overlap_size))
        x_shape[1] = X_test_raw.shape[1] * window_size
        X_test = np.ndarray(x_shape, dtype=float)
        Y_test = np.ndarray(x_shape[0])
        
#         if self.extra_FE_params is None:
#             X_test, num_of_added_samples = self.feature_extractor( X_test_raw, window_size, window_overlap_size)
#         else:
        X_test, num_of_added_samples = self.feature_extractor( X_test_raw, window_size, window_overlap_size, fe_params)
        Y_test = Y_test_raw[window_size-1:]
        self.logging.info('Done extracting features from test data! X_test.shape is %s and Y_test.shape is: %s', str(len(X_test)), str(len(Y_test)))
        return X_test, Y_test

    def prepare_samples_async(self, subject_number, raw_X, raw_Y, params_dict, fe_params):    
        """Throw away the first samples and the last samples from movement and NC and then generate data samples for the classifier.
        In asynchronous datasets, I assumed that each trial begins with the movement start so I was able to concatenate the  NC parts
        of two consecutive trials. This way feature extraction was easier."""
        
#         params_list = map(int,params_list)
        X_new = []
        Y_new = []    
        discard_mv_begin = params_dict['discard_mv_begin']#params_list[0]
        discard_mv_end = params_dict['discard_mv_end']#params_list[1]
        discard_nc_begin = params_dict['discard_nc_begin']#params_list[2]
        discard_nc_end = params_dict['discard_nc_end']#params_list[3]
        window_size = int(params_dict['window_size'])#params_list[4]
        window_overlap_size = int(params_dict['window_overlap_size'])#params_list[5]
        
        
        if 'movement_start_list' in self.config.configuration.keys():
            
            trial_size = self.config.configuration['trial_size']
             
            mv_start = self.config.configuration['movement_start_list'][subject_number]
            mv_trials_begin = np.array(range(int(self.config.configuration['number_of_all_trials_list'][subject_number]))) * self.config.configuration['trial_size'] + mv_start 
         
            nc_trials_begin = mv_trials_begin + self.config.configuration['movement_trial_size_list'][subject_number]
        else:
            mv_trials_begin, nc_trials_begin = self.calc_nc_trials_begin(raw_X, raw_Y)
        
#         mv_start = self.config.configuration['movement_start_list'][subject_number]
#         mv_trials_begin1 = np.array(range(int(self.config.configuration['number_of_all_trials_list'][subject_number]))) * self.config.configuration['trial_size'] + mv_start 
#         
#         nc_trials_begin1 = mv_trials_begin + self.config.configuration['movement_trial_size_list'][subject_number]
#         mv_trials_begin2, nc_trials_begin2 = self.calc_nc_trials_begin(raw_X, raw_Y)

        
        for mv_trail_begin_ind, mv_trial_begin in enumerate(mv_trials_begin):
            nc_trial_begin = nc_trials_begin[mv_trail_begin_ind]
            if not 'movement_start_list' in self.config.configuration.keys():
                if mv_trail_begin_ind != (len(mv_trials_begin) - 1): 
                    trial_size = mv_trials_begin[mv_trail_begin_ind+1] - mv_trial_begin
                else:
                    trial_size = raw_Y[-1] - mv_trial_begin
            
            trail_end = mv_trial_begin + trial_size
            
            #add new movement data sample
            mv_sample = raw_X[int(mv_trial_begin + discard_mv_begin):int(nc_trial_begin - discard_mv_end),:]
            X_new, Y_new = self.extract_feature_each_window( mv_sample, X_new, Y_new, window_size, window_overlap_size, raw_Y[mv_trial_begin + 1], fe_params)
            
            #add new NC data samples if the problem is self-paced
            nc_sample = raw_X[(nc_trial_begin + discard_nc_begin):(trail_end - discard_nc_end),:]
            X_new, Y_new = self.extract_feature_each_window( nc_sample, X_new, Y_new, window_size, window_overlap_size, raw_Y[nc_trial_begin + 1], fe_params)
        
        return X_new, Y_new

    def calc_nc_trials_begin(self, raw_X, raw_Y):
        pass
        mv_trials_begin = []
        nc_trials_begin = []
        last_label = raw_Y[0]
        if last_label != 1:
            mv_trials_begin.append(0)
        else:
            nc_trials_begin.append(0)
            
        for label_ind, label in enumerate(raw_Y):
            if label != last_label and label != 1:
                mv_trials_begin.append(label_ind)
            elif label != last_label and label == 1:
                nc_trials_begin.append(label_ind)
            last_label = label
            
        return mv_trials_begin, nc_trials_begin     
        

    def prepare_samples_sync(self, subject_number, raw_X, raw_Y, params_dict, fe_params):    
        """Throw away the first samples and the last samples from movement and NC and then generate data samples for the classifier"""
        mv_start = self.config.configuration['movement_start_list'][subject_number]
        
#         params_list = map(int,params_list)
        X_new = []
        Y_new = []    
        discard_mv_begin = params_dict['discard_mv_begin']#params_list[0]
        discard_mv_end = params_dict['discard_mv_end']#params_list[1]
        discard_nc_begin = params_dict['discard_nc_begin']#params_list[2]
        discard_nc_end = params_dict['discard_nc_end']#params_list[3]
        window_size = int(params_dict['window_size'])#params_list[4]
        window_overlap_size = int(params_dict['window_overlap_size'])#params_list[5]
        
        trial_size = self.config.configuration['trial_size']
        num_trials = len(raw_X) / trial_size 

        mv_trials_begin = np.array(range(num_trials)) * self.config.configuration['trial_size'] + mv_start 
        mv_size = self.config.configuration['movement_trial_size_list'][subject_number]
        
        nc_trials_begin = mv_trials_begin + self.config.configuration['movement_trial_size_list'][subject_number] - self.config.configuration['trial_size']
        
        for mv_trail_begin_ind, mv_trial_begin in enumerate(mv_trials_begin):
#             print mv_trail_begin_ind
            nc_trial_begin = nc_trials_begin[mv_trail_begin_ind]
            trail_end = nc_trials_begin[mv_trail_begin_ind] + trial_size
            
            #add new movement data sample
            mv_sample = raw_X[(mv_trial_begin + discard_mv_begin):(mv_trial_begin + mv_size - discard_mv_end),:]
            if window_size == -1:
                window_size = len(mv_sample)
                window_overlap_size = len(mv_sample) 
            X_new, Y_new = self.extract_feature_each_window( mv_sample, X_new, Y_new, window_size, window_overlap_size, raw_Y[mv_trial_begin + 1], fe_params)
            
        return X_new, Y_new

    def extract_feature_each_window(self, sample, X_new, Y_new, window_size, window_overlap_size, label, fe_params):
        """    """
        assert len(sample.shape) == 2
        
        X_new_ex, num_of_added_samples = self.feature_extractor( sample, window_size, window_overlap_size, fe_params)
        X_new = X_new + X_new_ex
        for i in range(num_of_added_samples): 
            Y_new.append(label)  
        return X_new, Y_new       
        
    def apply_CSP(self, X, raw_Y):
        """   """
#       TODO: first normalize data
        scaler = StandardScaler()

        X = scaler.fit_transform(X) 
        
        labels = list(set(raw_Y[np.logical_not(np.isnan(raw_Y))]))
        if self.config.configuration["dataset_type_str"] == 'sync':
            labels.remove(0)
        
        
        nchoose2 = list(itertools.combinations(labels,2))
        cov_mats = np.zeros(shape=(len(labels), int(X.shape[1]),int(X.shape[1])))
        
        for ind, label in enumerate(labels):
            cov_mats[ind,:,:] = np.dot(X[raw_Y == label].T,X[raw_Y == label])
            
        m = self.number_of_CSPs
        csp_res = np.zeros(shape = (m*len(nchoose2), X.shape[1]))
        
        for i in range(len(nchoose2)):
            
            squeezed_cov1 = np.squeeze(cov_mats[labels.index(nchoose2[i][0]),:,:])
            squeezed_cov2 = np.squeeze(cov_mats[labels.index(nchoose2[i][1]),:,:])
            csp_res[(i*m):((i+1)*m),:] = csp(squeezed_cov1, squeezed_cov2, m)
        
        return np.dot(X, csp_res.T)
        
        
        
############################################ The following methods have not been evaluated yet
    
    def extract_data_samples_test(self, Y_new, X_new, Y_test_raw, sample, window_size, window_overlap_size):
        """extract data samples for the classifier by using overlapping windows"""
        first_indices_of_data_sample = range(0, sample.shape[0] - window_size + 1, window_overlap_size)
        
        for i, first_index in enumerate(first_indices_of_data_sample):
            new_sample = np.reshape(sample[first_index:(first_index+window_size),:], window_size * sample.shape[1]) # TODO:
            X_new[i,:] = new_sample
            Y_new[i] = Y_test_raw[first_index]
#            print(i)
            
        return Y_new, X_new

    def prepare_test_data(self, subject, params_dict):
        """   """
        
        window_size = int(params_dict['window_size'])#int(params_list[-2])
        window_overlap_size = 1 #int(params_list[-1])
        
        X_test_raw, Y_test_raw = self.load_dataset_test(subject)
        x_shape = [0,0]
        x_shape[0] = len(range(0, len(X_test_raw) - window_size + 1, window_overlap_size))
        x_shape[1] = X_test_raw.shape[1] * window_size
        X_test = np.ndarray(x_shape, dtype=float)
        Y_test = np.ndarray(x_shape[0])
        
        Y_test, X_test = self.extract_data_samples_test(Y_test, X_test, Y_test_raw, X_test_raw, window_size, window_overlap_size) # , 1 )
        
        return X_test, Y_test
