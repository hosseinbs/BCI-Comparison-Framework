import Configuration_BCI
import RandomForest_BCI
import os
import Single_Job_runner as SJR
import Data_Preprocessor as Data_Proc 
import numpy as np
import logging 
import re
import time 
from subprocess import call

class Main:
    """this is the main class to run classification or regression on the BCI dataset"""
           
    def __init__(self, dir, dataset_name, learner_name,feature_extractor_name, channels = 'ALL', number_of_CSPs = -1, python_path = 'python'):
        """"""
        self.dir = dir
        self.dataset_name = dataset_name
        self.config = Configuration_BCI.Configuration_BCI(dir, self.dataset_name)#, channels)
        self.config.set_channel_type(channels, number_of_CSPs)
        self.logging = logging
        
        if self.config.configuration['logging_level_str'] == 'INFO':
            self.logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
        else:
            self.logging.basicConfig(level=logging.CRITICAL)
            
        self.learner_name = learner_name
        self.feature_extractor_name = feature_extractor_name
        self.channels = channels
        self.number_of_CSPs = number_of_CSPs
        self.myPython_path = python_path
        
        self.results_path = self.config.configuration['results_path_str']
        self.opt_results_path = self.config.configuration['results_opt_path_str']   
        
        self.bash_script = ["#!/bin/bash\n","#PBS -S /bin/bash\n","#PBS -M hosseinbs@gmail.com\n","#PBS -m bea\n","#PBS -l mem=4gb\n",
                            "module load application/python/2.7.3\n","module load python/2.7.5.anaconda\n",
                            "module load python/2.7.3\n",'cd $PBS_O_WORKDIR' ,'\n echo "Current working directory is `pwd`"\n', 'echo "Starting: run at: `date`"\n',
                             self.myPython_path +  " ./BCI_Framework/Single_Job_runner.py", 'echo "Program finished with exit code $? at: `date`"\n']


        self.python_run_cmd_ind = 11
        self.set_results_path()
        
        self.logging.info('dataset: %s learning algorithm: %s feature extraction method: %s channels: %s Number of CSPs: %s python path: %s' +
        ' results path: %s optimal results path: %s', self.dataset_name, self.learner_name, self.feature_extractor_name, self.channels,
        self.number_of_CSPs, self.myPython_path, self.results_path, self.opt_results_path)
                
                
    def run_learner_gridsearch(self):
        """perform grid search in the hyper-parameters space and submit jobs"""
        
        for subject in self.config.configuration['subject_names_str']:
            
            discard_mvs_begin = self.config.configuration['discarded_samples_begining_movement_list']
            discard_ncs_begin = self.config.configuration['discarded_samples_begining_nc_list']
            discard_mvs_end = self.config.configuration['discarded_samples_end_movement_list']
            discard_ncs_end = self.config.configuration['discarded_samples_end_nc_list']
            window_sizes = self.config.configuration['windowSize_crossval_list']
            window_overlap_sizes = self.config.configuration['window_overlap_size_crossval_list']

            if not (self.feature_extractor_name in self.config.configuration['fe_params_dict'].keys()):
                fe_params = None
            else:
                fe_params = self.config.configuration['fe_params_dict'][self.feature_extractor_name]
                
            for discard_index_mv, discard_mv in enumerate(discard_mvs_begin):
                for discard_index_nc, discard_nc in enumerate(discard_ncs_begin):
                    for win_ind, window_size in enumerate(window_sizes): 
                        
                        window_overlap_size =  window_overlap_sizes[win_ind]
                        
                        params_dict = {'discard_mv_begin' : discard_mv, 'discard_mv_end' : discard_mvs_end[discard_index_mv], 
                                       'discard_nc_begin' : discard_nc, 'discard_nc_end' : discard_ncs_end[discard_index_nc],
                                       'window_size' : window_size, 'window_overlap_size' : window_overlap_size,
                                       'fe_params': fe_params, 'channel_type':self.config.channel_type + "NCSPS" +  str(self.number_of_CSPs)}
                        
                        self.logging.info('job parameters: %s', params_dict)
                        res = SJR.Simple_Job_Runner.check_if_job_exists(self.results_path, subject, params_dict)
                        if not res:
                            self.logging.info('begin submitting job!')
                            self.submit_train_learner(subject, params_dict)
                        else:
                            self.logging.info('job exists!!!')


    def write_feature_matrices_gridsearch(self):

        for subject in self.config.configuration['subject_names_str']:
            
            discard_mvs_begin = self.config.configuration['discarded_samples_begining_movement_list']
            discard_ncs_begin = self.config.configuration['discarded_samples_begining_nc_list']
            discard_mvs_end = self.config.configuration['discarded_samples_end_movement_list']
            discard_ncs_end = self.config.configuration['discarded_samples_end_nc_list']
            window_sizes = self.config.configuration['windowSize_crossval_list']
            window_overlap_sizes = self.config.configuration['window_overlap_size_crossval_list']

            if not (self.feature_extractor_name in self.config.configuration['fe_params_dict'].keys()):
                fe_params = None
            else:
                fe_params = self.config.configuration['fe_params_dict'][self.feature_extractor_name]
                
            for discard_index_mv, discard_mv in enumerate(discard_mvs_begin):
                for discard_index_nc, discard_nc in enumerate(discard_ncs_begin):
                    for win_ind, window_size in enumerate(window_sizes): 
                        
                        window_overlap_size =  window_overlap_sizes[win_ind]
                        
                        params_dict = {'discard_mv_begin' : discard_mv, 'discard_mv_end' : discard_mvs_end[discard_index_mv], 
                                       'discard_nc_begin' : discard_nc, 'discard_nc_end' : discard_ncs_end[discard_index_nc],
                                       'window_size' : window_size, 'window_overlap_size' : window_overlap_size,
                                       'fe_params': fe_params, 'channel_type':self.config.channel_type + "NCSPS" + str(self.number_of_CSPs)}
                        
                        self.logging.info('job parameters: %s', params_dict)
                        res = SJR.Simple_Job_Runner.check_if_job_exists(self.feature_matrix_path, subject, params_dict)
                        if not res:
                            job_runner = SJR.Single_Job_Runner(self.dir, self.learner_name, self.feature_extractor_name,self.dataset_name, subject, params_dict.copy())
                            job_runner.save_feature_matrices()
                        else:
                            self.logging.info('feature file exists!!!')

    def submit_train_learner(self, subject, params_dict, optimal = False):
        """submits a job to train the learner"""

        subject = subject.replace('\n','')
        if os.name == 'nt':
            self.logging.info('begin running job on linux')
            
            temp = self.bash_script[self.python_run_cmd_ind]
            if type(params_dict['fe_params']) is list:
                self.bash_script[self.python_run_cmd_ind] = self.bash_script[self.python_run_cmd_ind] + " " + self.dir + " " + self.learner_name + " " + self.feature_extractor_name + " "+ self.dataset_name + " " + subject + " " + str(optimal) +  " " + ' '.join(['%s=%s' % (key, value) for (key, value) in params_dict.items() if key != 'fe_params']) + ' fe_params=' + ','.join(str(e) for e in params_dict['fe_params'])+ '\n'
            else:
                self.bash_script[self.python_run_cmd_ind] = self.bash_script[self.python_run_cmd_ind] + " " + self.dir + " " + self.learner_name + " " + self.feature_extractor_name + " "+ self.dataset_name + " " + subject + " " + str(optimal) +  " " + ' '.join(['%s=%s' % (key, value) for (key, value) in params_dict.items()]) + '\n'
                    
            with open('myRun.pbs', 'w') as my_run_file:
                my_run_file.writelines(self.bash_script)
            
            job_name = ''.join(['%s_' %(value) for (key, value) in params_dict.items() if key != 'fe_params']) + subject + self.feature_extractor_name + self.learner_name
            print job_name
            job_name = str(hash(job_name)%(10**15)) 

            submit_command = "qsub -l walltime=" + str(self.config.configuration["durations_dict"][self.learner_name]) + ":00:00 -N " + job_name  + ' ' + my_run_file.name 

            #################################################### check if job is already submitted, do not submit it again            
            command = 'qstat -u hosseinb'
            p = os.popen( command,"r")
            submitted_jobs = p.readlines()[5:]
            current_already_submitted = False
            
            for submitted_job in submitted_jobs:
                if job_name in submitted_job:
                    current_already_submitted = True
                    print 'job already exists'
                    break
            
            if not current_already_submitted:
                try :
                    self.logging.info('command to submit the job:\n %s', submit_command)
                    os.system(submit_command)
                    self.logging.info('job submitted without error!!!')
    
                except:
                    self.logging.info('exception in job submission')

            time.sleep(2)

            self.bash_script[self.python_run_cmd_ind] = temp 
                      
        elif os.name == 'posix':
            
            self.logging.info('begin running job on windows')
            job_runner = SJR.Single_Job_Runner(self.dir, self.learner_name, self.feature_extractor_name,self.dataset_name, subject, params_dict.copy(),optimal)
            job_runner.run_learner()
        
    def set_results_path(self):
        """set the results paths for both the cross validation and optimal learner"""
        
        self.results_path, self.opt_results_path = SJR.Single_Job_Runner.set_results_path(self.results_path, self.opt_results_path, self.learner_name, self.feature_extractor_name)
        self.feature_matrix_path = SJR.Single_Job_Runner.set_feature_matrix_path(self.config.configuration['feature_matrix_dir_name_str'], self.feature_extractor_name)
        
        
    def find_learners_optimal_params(self, subject):
        """This function reads the results files and finds the optimal values for the learner"""
        best_score = np.Inf
        best_score_file = ''
        self.logging.info('searching for best parameters for subject: %s in folder: %s', subject, self.results_path)
        file_names = [ f for f in os.listdir(self.results_path) if os.path.isfile(os.path.join(self.results_path,f)) ]
        
        best_learner_params = {}
        for file_name in file_names:
            if file_name[-len(subject):] == subject:
                         
                job_runner = SJR.Simple_Job_Runner(self.dir, self.learner_name, self.feature_extractor_name,self.dataset_name)
                current_error, learner_params = job_runner.my_Learner_Manager.find_cv_error(os.path.join(self.results_path,file_name))
                if current_error < best_score:
                    best_score_file = file_name
                    best_score = current_error 
                    best_learner_params = learner_params
        
        self.logging.info('best_score is: %s', str(best_score))            
        best_score_file = best_score_file[:-len('_' + subject)] 
#        best_score_file.replace('_' + subject, '')

        channel_params = best_score_file.split('_')[-1]
        params = best_score_file.split('_')[0:-1]
        return best_learner_params, map(float, params), channel_params, best_score
    
            
    def test_learner(self):
        """ """
        self.logging.info("started finding the optimal parameters")
        for subject in self.config.configuration['subject_names_str']:
            best_learner_params, params_list, channel_params, best_score_temp = self.find_learners_optimal_params(subject)
            
            params_dict = {'discard_mv_begin' : params_list[-6], 'discard_mv_end' : params_list[-5],
                                          'discard_nc_begin' : params_list[-4], 'discard_nc_end' : params_list[-3],
                                       'window_size' : params_list[-2], 'window_overlap_size' : params_list[-1], 'channel_type': channel_params}
            
            self.logging.info('optimal parameters are: ' + str(params_dict))
            
            self.logging.info("submitting learner with optimal parameters!")
            res = SJR.Simple_Job_Runner.check_if_job_exists(self.opt_results_path, subject + '.npz', params_dict)
            if not res:
                self.logging.info('Submitting optimal job started!')
                self.submit_train_learner(subject, dict(params_dict.items() + best_learner_params.items() + {'best_error':best_score_temp}.items()), optimal = True)
            else:
                self.logging.info('Optimal job exists!!!')
                
            

    def run_learner_BO(self, subject, params_dict):
        """ """
        res = SJR.Single_Job_Runner.check_if_job_exists(self.results_path, subject, params_dict)
        if not res:
            self.logging.info('begin submitting job!')
            self.submit_train_learner(subject, params_dict.copy())
            return False
        
        return True
        
    def run_optimal_learner_BO(self, subject, params_dict):
        """ """
#        I should submit the job but as the running time is fast I calculate the results using the windows method
#        self.submit_train_learner(subject, params_dict, optimal = True)
        res = SJR.Single_Job_Runner.check_if_job_exists(self.opt_results_path, subject + '.npz', params_dict)
        if not res:
            self.logging.info('Submitting optimal job started!')
            
            self.submit_train_learner(subject, params_dict.copy(), optimal = True)
            
        return res

    def find_opt_error(self, subject):
        
        file_names = [ f for f in os.listdir(self.opt_results_path) if os.path.isfile(os.path.join(self.opt_results_path,f)) ]
        
        
        for file_name in file_names:
            
            underline_indices = [m.start() for m in re.finditer("_", file_name)]
            current_subj = file_name[underline_indices[-1]+1:]

            if current_subj == subject:
                with open(os.path.join(self.opt_results_path,file_name),'r') as res_file:
                    current_acc = float(res_file.readline())
                    return current_acc
                
