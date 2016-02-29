import os
import re
import numpy as np
import ast
import itertools

class Configuration_BCI:
    """This class contains the settings for the BCI classification task, the data used here has been preprocessed before!
    """
    
    def __init__(self, dir, dataset_name):#, channels = 'ALL'):
        
        self.dataset_name = dataset_name
        self.configuration = {}
        config_file_name = dataset_name + '_spec'
        config_file_name = os.path.join(dir, config_file_name)
        
#         print os.getcwd()
        with open(config_file_name, 'r') as f:
            config_data = f.readlines()
            
        for c in config_data:
            [key, value] = c.split('==')
            if key.find('_str') != -1:#if string
                value = value.split(",")
#                value[-1] = value[-1].replace('\n','')
                if len(value) == 1:
                    value[0] = value[0].rstrip('\r\n')
                    value = value[0]
                else:
                    value = [val.rstrip('\r\n') for val in value]
                self.configuration[key] = value
            elif key.find('_list') != -1:#if list
#                print value
#                value = [v.rstrip('\r\n') for v in value]
                value = [int(s) for s in value.split(",")]
                self.configuration[key] = value
            elif key.find('_dict') != -1:#
                self.configuration[key] = ast.literal_eval(value)
            else:#if scalar
#                value = value.rstrip('\r\n')
                value = [int(s) for s in value.split(",")]
                if len(value) == 1:
                    value = value[0]
                self.configuration[key] = value
        
        channel_names = np.array(self.configuration['channel_names_str'])
#             
        self.test_config()
    
    
    def set_channel_type(self, channels, n_csps):
        """  """
        channel_names = np.array(self.configuration['channel_names_str'])
        self.channel_type = channels
        
        if channels == 'ALL' or channels == 'CSP':
            self.which_channels = range(self.configuration['number_of_channels'])
#            print(channel_names[self.which_channels])
            
        if channels == 'CS':
            
            self.which_channels =  map(lambda x: True if re.match('^C\w$', x) != None else False,channel_names)
            self.configuration['number_of_channels'] = sum(self.which_channels) 
            
#            print(channel_names[np.array(self.which_channels)])
        elif channels == 'C34':
            
            self.which_channels =  map(lambda x: True if re.match('C3|C4', x) != None else False,channel_names)
            self.configuration['number_of_channels'] = sum(self.which_channels)
        
        elif channels == 'C34Z':
            
            self.which_channels =  map(lambda x: True if re.match('C3|C4|Cz', x) != None else False,channel_names)
            self.configuration['number_of_channels'] = sum(self.which_channels)
        elif channels == 'CSP':
            self.configuration['number_of_channels'] = len(list(itertools.combinations(range(self.configuration['number_of_classes']),2))) * int(float(n_csps))
            pass
        elif channels.isdigit():
            binary_str = bin(int(channels))[2:]
            self.which_channels =  map(lambda x: True if x == '1' else False,binary_str)
            dif_n_channels =  self.configuration['number_of_channels'] - len(self.which_channels)
            if dif_n_channels > 0:
                self.which_channels = [False] * dif_n_channels + self.which_channels
            self.configuration['number_of_channels'] = sum(self.which_channels)
        
    
    
    def test_config(self):
        pass
#        print(self.configuration)
#        assert len(self.configuration['windowSize_crossval']) == len(self.configuration['window_overlap_size_crossval'])
#        assert len(self.configuration['discarded_samples_begining_nc']) == len(self.configuration['discarded_samples_end_nc'])
#        assert len(self.configuration['discarded_samples_begining_movement']) == len(self.configuration['discarded_samples_end_movement'])
#         print(self.dataset_name)

