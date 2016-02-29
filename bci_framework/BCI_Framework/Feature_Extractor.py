import Data_Preprocessor as DP
import Filter as filter
import numpy as np
from math import sqrt, pi, log
from scipy import linalg as LA
from morlet_wavelet import *
import matplotlib
matplotlib.use('Agg')
from spectrum import *

class Feature_Extractor:
    """ """
    
    config = None
    
    def __init__(self, dir, dataset_name, params_list):
        """ """
        dp = DP.Data_Preprocessor(dir, dataset_name)
        params_list = map(float,params_list)
        X_new = []
        Y_new = []    
        discard_mv_begin = params_list[0]
        discard_mv_end = params_list[1]
        discard_nc_begin = params_list[2]
        discard_nc_end = params_list[3]
        window_size = params_list[4]
        window_overlap_size = params_list[5]
        cutoff_freq_low = params_list[6]
        cutoff_freq_high = params_list[7]

    
    @staticmethod
    def bp_feature_extractor(sample, window_size, window_overlap_size, fe_params = None):
        """this method extracts Band power features from the given movement or NC trial, and 
        returns the feature vectors and the number of added data samples"""
        X_new = []
        if window_overlap_size != 0:
            first_indices_of_data_sample = range(0, int(sample.shape[0] - window_size + 1), int(window_overlap_size))
        else:
            first_indices_of_data_sample = [0]
            
        for first_index in first_indices_of_data_sample:
#            test = sample[first_index:(first_index+window_size),:] * weights
            new_sample = np.diag(np.dot(sample[first_index:(first_index+window_size),:].T , sample[first_index:(first_index+window_size),:]))
#            new_sample = np.diag(np.dot(test.T , test))
            X_new.append(np.copy(new_sample))
            
        return X_new, len(first_indices_of_data_sample)
    
    
    @staticmethod
    def logbp_feature_extractor(sample, window_size, window_overlap_size, fe_params = None):
        """this method extracts Band power features from the given movement or NC trial, and 
        returns the feature vectors and the number of added data samples"""
        X_new = []
        if window_overlap_size != 0:
            first_indices_of_data_sample = range(0, int(sample.shape[0] - window_size + 1), int(window_overlap_size))
        else:
            first_indices_of_data_sample = [0]
            
        for first_index in first_indices_of_data_sample:
#            test = sample[first_index:(first_index+window_size),:] * weights
            new_sample = np.log(np.diag(np.dot(sample[first_index:(first_index+window_size),:].T , sample[first_index:(first_index+window_size),:])))
#            new_sample = np.diag(np.dot(test.T , test))
            X_new.append(np.copy(new_sample))
            
        return X_new, len(first_indices_of_data_sample)
    
    
    @staticmethod
    def wackerman_feature_extractor(sample, window_size, window_overlap_size, fe_params = None):
        """ """
        
#     TODO: nomalize data before calling this method
        sampling_rate = Feature_Extractor.config.configuration['sampling_rate']
        X_new = []
        first_indices_of_data_sample = range(0, int(sample.shape[0] - window_size + 1), int(window_overlap_size))
        for first_index in first_indices_of_data_sample:
            
            new_sample = np.copy(sample[first_index:(first_index+window_size),:])

            N = new_sample.shape[0] # number of samples
            K = new_sample.shape[1] # number of electrodes
            
            for i in range(K): #Zero mean in each channel
                new_sample[:,i]=new_sample[:,i]-np.mean(new_sample[:,i]);

            for i in range(N): #Common average reference for each sample
                new_sample[i,:]=new_sample[i,:]-np.mean(new_sample[i,:]);

    
            d0 = np.sum((new_sample)**2)
            d1 = np.sum((new_sample[1:len(new_sample)] - new_sample[0:len(new_sample)-1])**2)
    
            m0 = d0 / N
            m1 = d1 * (sampling_rate**2) / (N-1)
            
            SIGMA = sqrt(m0/K)
            PHI   = sqrt(m1/m0)/(2*pi)
            
            covMat = np.dot(sample.T, sample) / N
            
            e_vals, e_vecs = LA.eig(covMat)
            e_vals = np.real(e_vals / np.sum(e_vals))
    
            logOmega = -np.dot(np.log(e_vals), e_vals)
            X_new.append([SIGMA, PHI, np.exp(logOmega)])
            
    
        return X_new, len(first_indices_of_data_sample)
    
    @staticmethod
    def raw_feature_extractor(sample, window_size, window_overlap_size, fe_params = None):
        """extract features from the given mv or nc trial, and retruns the feature vectors and the number of added data samples"""
        X_new = []
        first_indices_of_data_sample = range(0, int(sample.shape[0] - window_size + 1), int(window_overlap_size))
        for first_index in first_indices_of_data_sample:
            new_sample = np.reshape(sample[first_index:(first_index+window_size),:], window_size * sample.shape[1]) # TODO:
            X_new.append(np.copy(new_sample))
            
        return X_new, len(first_indices_of_data_sample)

    
    @staticmethod
    def morlet_feature_extractor(sample, window_size, window_overlap_size, fe_params = None):
        """this method extracts Band power features from the given movement or NC trial, and 
        returns the feature vectors and the number of added data samples"""
        fvec = range(4,31)
        Fs = float(Feature_Extractor.config.configuration['sampling_rate'])
        n_channels = Feature_Extractor.config.configuration['number_of_channels']
        nfreq = len(fvec)-1;

        X_new = []
        
        if window_overlap_size != 0:
            first_indices_of_data_sample = range(0, sample.shape[0] - window_size + 1, window_overlap_size)
        else:
            first_indices_of_data_sample = [0]
            
        for first_index in first_indices_of_data_sample:
            new_sample = np.zeros(n_channels * nfreq) + np.inf
            
            Nx = sample[first_index:(first_index+window_size),:].shape[0]
            for channel_ind in range(n_channels):
                tfr,t,f, wt = morlet_wavelet_transform(sample[first_index:(first_index+window_size),channel_ind],np.arange(Nx), sqrt(Nx),fvec[0]/Fs,fvec[-1]/Fs,128)
                margt, margf, E = margtfr(np.array(tfr))
                f = np.array([Fs*abs(i) for i in f])

                # now simply pick up the energy in each band
                for k in range(nfreq):
                    new_sample[channel_ind * nfreq + k] = log(np.sum( margf[np.logical_and(np.greater_equal(f,fvec[k]), np.less(f,fvec[k+1]))] ))

            X_new.append(np.copy(new_sample))
            
        return X_new, len(first_indices_of_data_sample)
    
    
    @staticmethod
    def ar_feature_extractor(sample, window_size, window_overlap_size, fe_params = None):
        """this method extracts Auto Regressive features from the given movement or NC trial, and 
        returns the feature vectors and the number of added data samples"""
        X_new = []
        if window_overlap_size != 0:
            first_indices_of_data_sample = range(0, sample.shape[0] - window_size + 1, window_overlap_size)
        else:
            first_indices_of_data_sample = [0]
            
        for first_index in first_indices_of_data_sample:
            new_sample = np.zeros(sample.shape[1] * fe_params)
            for channel in range(sample.shape[1]):
#            test = sample[first_index:(first_index+window_size),:] * weights
                ar, variance, coeff_reflection = aryule(sample[first_index:(first_index+window_size),channel], fe_params)
                
                new_sample[channel*fe_params:(channel+1)*fe_params] = np.array(coeff_reflection)
#            new_sample = np.diag(np.dot(test.T , test))
            X_new.append(np.copy(new_sample))
            
        return X_new, len(first_indices_of_data_sample)
    