import os
import numpy as np
import random
import scipy.io as sio
import re
import pickle 

def FACED_old2new(old_dir,new_dir):
    """Convert old FACED dataset to new format."""
    # input data shape(onesubsession):(vid,channel,points_30s)
    # output data shape:              (channel,points_allvids)
    list_files = os.listdir(old_dir)
    list_files = sorted(list_files, key=lambda x: int(re.search(r'\d+', x).group()))
    os.makedirs(new_dir, exist_ok=True)
    
    for idx,fn in enumerate(list_files):
        file_path = os.path.join(old_dir,fn)
        
        with open(file_path, 'rb') as fo:     # 读取pkl文件数据
            onesub_data = pickle.load(fo, encoding='bytes')
            new_data = np.transpose(onesub_data, (1, 0, 2)).reshape(onesub_data.shape[1], -1)
            print(new_data.shape)
            
            
            n_samples_one = [[30]*onesub_data.shape[0]]
            print(n_samples_one)
            
            base_name, extension = os.path.splitext(fn)
            new_fn = os.path.join(new_dir, f'{base_name}.mat')
            print(new_fn)
            
            sio.savemat(new_fn, {'data_all_cleaned': new_data,'n_samples_one':n_samples_one})
            print()
            
if __name__ == '__main__':
    old_dir = '/mnt/data/model_weights/grm/FACED/processed_data/'
    new_dir = '/mnt/data/model_weights/grm/FACED_old/processed_data/'
    FACED_old2new(old_dir,new_dir)