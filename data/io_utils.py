import os
import numpy as np
import scipy.io as sio
import re

def get_load_data_func(dataset_name):
    if dataset_name == 'SEEDV':
        return load_processed_SEEDV_NEW_data
    elif dataset_name == 'SEED':
        return load_processed_SEED_NEW_data
    elif dataset_name == 'FACED':
        return load_processed_FACED_NEW_data
    else:
        raise ValueError('dataset_name not found')

def load_EEG_data(data_dir, cfg):
    load_data_func = get_load_data_func(cfg.dataset_name)
    data, onesub_labels, n_samples_onesub, n_samples_sessions = load_data_func(
                                data_dir, cfg.fs, cfg.n_channs, cfg.timeLen, cfg.timeStep, 
                                cfg.n_session, cfg.n_subs, cfg.n_vids, cfg.n_class)
    return data, onesub_labels, n_samples_onesub, n_samples_sessions

def load_finetune_EEG_data(data_dir, cfg):
    load_data_func = get_load_data_func(cfg.dataset_name)
    data, onesub_labels, n_samples_onesub, n_samples_sessions = load_data_func(
                                data_dir, cfg.fs, cfg.n_channs, cfg.timeLen2, cfg.timeStep2, 
                                cfg.n_session, cfg.n_subs, cfg.n_vids, cfg.n_class)
    return data, onesub_labels, n_samples_onesub, n_samples_sessions


def load_processed_FACED_NEW_data(dir, fs, n_chans, timeLen,timeStep,n_session=1, 
                                  n_subs=123, n_vids = 28, n_class=9, t=30):
    # t: we only use last 30s data for each video
    # input data shape(onesub):(vid,channel,time)
    # output : (subs*slices*vids)*channals*time
    #           (123*28*30)*30*(125)

    list_files = os.listdir(dir)
    list_files = sorted(list_files, key=lambda x: int(re.search(r'\d+', x).group()))
    assert len(list_files) == n_subs
    n_samples = int((t-timeLen)/timeStep)+1
    points_len = int(timeLen*fs)
    points_step = int(timeStep*fs)

    if n_class == 2:
        vid_sel = list(range(12))
        vid_sel.extend(list(range(16,28)))
        # data = data[:, vid_sel, :, :] # sub, vid, n_channs, n_points
        n_vids = 24
    elif n_class == 9:
        vid_sel = list(range(28))
        n_vids = 28
    data = np.empty((n_subs,n_vids*n_samples,n_chans,fs*timeLen),float)
    # subs(*slices*vids)*channals*time

    for idx,fn in enumerate(list_files):
        file_path = os.path.join(dir,fn)
        onesubsession_data = sio.loadmat(file_path)
        
        EEG_data = onesubsession_data['data_all_cleaned']
        thr = 30 * np.median(np.abs(EEG_data))
        EEG_data = (EEG_data - np.mean(EEG_data[np.abs(EEG_data)<thr])) / np.std(EEG_data[np.abs(EEG_data)<thr])
        n_points = onesubsession_data['n_samples_one'][0]*fs
        n_points_cum = np.cumsum(n_points).astype(int)
        start_points = n_points_cum-t*fs
        
        for k, vid in enumerate(vid_sel):
            for i in range(n_samples):
                data[idx,k*n_samples+i] = EEG_data[:,start_points[vid]+i*points_step:start_points[vid]+i*points_step+points_len]        
    
    data = data.reshape(-1,data.shape[-2],data.shape[-1])
    # (subs*slices*vids)*channals*time
    
    if n_class == 2:
        label = [0] * 12
        label.extend([1] * 12)
    elif n_class == 9:
        label = [0] * 3
        for i in range(1,4):
            label.extend([i] * 3)
        label.extend([4] * 4)
        for i in range(5,9):
            label.extend([i] * 3)
    
    onesub_labels = []
    for i in range(len(label)):
        onesub_labels = onesub_labels + [label[i]]*n_samples
        
    n_samples_onesub = np.array([n_samples]*n_vids)
    n_samples_sessions = n_samples_onesub.reshape(n_session,-1)

    return data, np.array(onesub_labels), n_samples_onesub, n_samples_sessions


def load_processed_SEEDV_data(dir, fs, n_chans, timeLen,timeStep, n_session, n_subs=16, n_vids = 15, n_class=5):
    # input data shape(onesub_onesession):(channels,tot_time) tot_time = sum(eachvids_n_points) 
    # output : (subs*sum(n_samples_onesub))*channals*time
    #           (15*(sum(n_samples_onesub)))*62*point_len(1250)

    list_files = os.listdir(dir)
    list_files.sort(key= lambda x:int(x[:-4]))
    # print(list_files)

    points_len = int(timeLen*fs)
    points_step = int(timeStep*fs)


    n_samples_onesub = []
    for i in range(n_session):
        fn = list_files[i]
        file_path = os.path.join(dir,fn)
        onesubsession_data = sio.loadmat(file_path)  
        n_points = np.squeeze(onesubsession_data['n_points']).astype(int)
        n_samples_onesubsession = ((n_points-points_len)//points_step+1).astype(int)
        n_samples_onesub = n_samples_onesub + list(n_samples_onesubsession)

    n_samples_sum_onesub = np.sum(n_samples_onesub)


    data = np.empty((n_subs*n_samples_sum_onesub,n_chans,points_len),float)

    s = np.arange(n_session)
    # n_samples_onesub = []
    cnt = 0
    for idx,fn in enumerate(list_files):
        file_path = os.path.join(dir,fn)
        # print(fn)
        onesubsession_data = sio.loadmat(file_path)     #keys: data,n_points
        EEG_data = onesubsession_data['data']   #(channels,tot_n_points)  (62,tot_n_points)
        thr = 30 * np.median(np.abs(EEG_data))
        EEG_data = (EEG_data - np.mean(EEG_data[EEG_data<thr])) / np.std(EEG_data[EEG_data<thr])
        n_points = np.squeeze(onesubsession_data['n_points']).astype(int)
        # print(EEG_data.shape)
        n_points_cum = np.concatenate((np.array([0]),np.cumsum(n_points)))
        n_samples_onesubsession = ((n_points-points_len)//points_step+1).astype(int)
        
        # if idx < n_session:
        #     if idx == s[idx]:
        #         n_samples_onesub = n_samples_onesub + list(n_samples_onesubsession)
        for vid in range(n_vids):
            # print('vid:',vid)
            for i in range(n_samples_onesubsession[vid]):
                # print('sample:',i)

                data[cnt] = EEG_data[:,n_points_cum[vid]+i*points_step:n_points_cum[vid]+i*points_step+points_len]
                cnt+=1

                # 拼接速度会越来越慢
                # temp = temp.reshape(1,temp.shape[0],temp.shape[1])
                # start_time = time.time()
                # data = np.concatenate((data,temp),0)
                # end_time = time.time()
                # print(end_time - start_time)
    # print(cnt)

    n_samples_onesub = np.array(n_samples_onesub)
    n_samples_sessions = n_samples_onesub.reshape(n_session,-1)
    label = [4, 1, 3, 2, 0] * 3 + [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0] * 2
    onesub_labels = []
    for i in range(len(label)):
        onesub_labels = onesub_labels + [label[i]]*n_samples_onesub[i]
    
    print('load processed data finished!')   

    return data, np.array(onesub_labels), n_samples_onesub, n_samples_sessions

def load_processed_SEEDV_NEW_data(dir, fs, n_chans, timeLen, timeStep, n_session=3, 
                                  n_subs=16, n_vids = 15, n_class=5):
    # input data shape(onesub_onesession):(channels,tot_time) tot_time = sum(eachvids_n_points) 
    # *input data shape（onesub_3session):(channels,tot_time)
    # output : (subs*sum(n_samples_onesub))*channals*time
    #           (16*(sum(n_samples_onesub)))*62*point_len(1250)
    

    list_files = os.listdir(dir)
    list_files = sorted(list_files, key=lambda x: int(re.search(r'\d+', x).group()))
    assert len(list_files) == n_subs
    points_len = int(timeLen*fs)
    points_step = int(timeStep*fs)
    
    # 3 session in all change delete the loop
    file_path = os.path.join(dir,list_files[0])
    onesub_data = sio.loadmat(file_path)  
    n_time = np.squeeze(onesub_data['merged_n_samples_one']).astype(int)
    n_points = np.array(n_time) * fs
    n_samples_onesub = ((n_points-points_len)//points_step+1).astype(int)
    n_samples_sum_onesub = np.sum(n_samples_onesub)
    
    data = np.empty((n_subs*n_samples_sum_onesub,n_chans,points_len),float)

    cnt = 0
    for idx,fn in enumerate(list_files):
        file_path = os.path.join(dir,fn)
        # print(fn)
        onesub_data = sio.loadmat(file_path)     #keys: data,n_points
        EEG_data = onesub_data['merged_data_all_cleaned']   #(channels,tot_n_points_3session)  (60,tot_n_points_3session)
        thr = 30 * np.median(np.abs(EEG_data))
        EEG_data = (EEG_data - np.mean(EEG_data[np.abs(EEG_data)<thr])) / np.std(EEG_data[np.abs(EEG_data)<thr])
        n_points_cum = np.concatenate((np.array([0]),np.cumsum(n_points)))

        
        n_vids_all = n_vids*n_session
        for vid in range(n_vids_all):
            # print('vid:',vid)
            for i in range(n_samples_onesub[vid]):
                # print('sample:',i)
                data[cnt] = EEG_data[:,n_points_cum[vid]+i*points_step:n_points_cum[vid]+i*points_step+points_len]
                cnt+=1
    
    n_samples_onesub = np.array(n_samples_onesub)
    n_samples_sessions = n_samples_onesub.reshape(n_session,-1)
    label = [4, 1, 3, 2, 0] * 3 + [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0] * 2
    onesub_labels = []
    for i in range(len(label)):
        onesub_labels = onesub_labels + [label[i]]*n_samples_onesub[i]   
    return data, np.array(onesub_labels), n_samples_onesub, n_samples_sessions

def load_processed_SEED_NEW_data(dir, fs, n_chans, timeLen, timeStep, n_session=3, 
                                  n_subs=15, n_vids = 15, n_class=3):
    # input data shape(onesub_onesession):(channels,tot_time) tot_time = sum(eachvids_n_points) 
    # *input data shape（onesub_3session):(channels,tot_time)
    # output : (subs*sum(n_samples_onesub))*channals*time
    #           (16*(sum(n_samples_onesub)))*62*point_len(1250)
    

    list_files = os.listdir(dir)
    list_files = sorted(list_files, key=lambda x: int(re.search(r'\d+', x).group()))
    assert len(list_files) == n_subs
    points_len = int(timeLen*fs)
    points_step = int(timeStep*fs)
    
    # 3 session in all change delete the loop
    file_path = os.path.join(dir,list_files[0])
    onesub_data = sio.loadmat(file_path)  
    n_time = np.squeeze(onesub_data['merged_n_samples_one']).astype(int)
    n_points = np.array(n_time) * fs
    n_samples_onesub = ((n_points-points_len)//points_step+1).astype(int)
    n_samples_sum_onesub = np.sum(n_samples_onesub)
    
    data = np.empty((n_subs*n_samples_sum_onesub,n_chans,points_len),float)

    cnt = 0
    for idx,fn in enumerate(list_files):
        file_path = os.path.join(dir,fn)
        # print(fn)
        onesub_data = sio.loadmat(file_path)     #keys: data,n_points
        EEG_data = onesub_data['merged_data_all_cleaned']   #(channels,tot_n_points_3session)  (60,tot_n_points_3session)
        thr = 30 * np.median(np.abs(EEG_data))
        EEG_data = (EEG_data - np.mean(EEG_data[np.abs(EEG_data)<thr])) / np.std(EEG_data[np.abs(EEG_data)<thr])
        n_points_cum = np.concatenate((np.array([0]),np.cumsum(n_points)))

        
        n_vids_all = n_vids*n_session
        for vid in range(n_vids_all):
            # print('vid:',vid)
            for i in range(n_samples_onesub[vid]):
                # print('sample:',i)
                data[cnt] = EEG_data[:,n_points_cum[vid]+i*points_step:n_points_cum[vid]+i*points_step+points_len]
                cnt+=1
    
    n_samples_onesub = np.array(n_samples_onesub)
    n_samples_sessions = n_samples_onesub.reshape(n_session,-1)
    label =  list(np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1])+1) * 3
    onesub_labels = []
    for i in range(len(label)):
        onesub_labels = onesub_labels + [label[i]]*n_samples_onesub[i]   
    return data, np.array(onesub_labels), n_samples_onesub, n_samples_sessions



def save_sliced_data(sliced_data_dir, data, onesub_labels, n_samples_onesub, n_samples_sessions):
    if not os.path.exists(sliced_data_dir+'/metadata'):
        os.makedirs(sliced_data_dir+'/metadata')
    if not os.path.exists(sliced_data_dir+'/data'):
        os.makedirs(sliced_data_dir+'/data')
    np.save(sliced_data_dir+'/metadata/onesub_labels.npy', onesub_labels)
    np.save(sliced_data_dir+'/metadata/n_samples_onesub.npy', n_samples_onesub)
    np.save(sliced_data_dir+'/metadata/n_samples_sessions.npy', n_samples_sessions)
    for sample in range(data.shape[0]):
        np.save(sliced_data_dir+f'/data/data_sample_{sample}.npy', data[sample])
    np.save(sliced_data_dir+'/saved.npy', [True])
    print('save sliced data finished!')

def test_load_processed_SEEDV_data():
    data_dir = '/mnt/data/model_weights/grm/SEEDV/EEG_processed_sxk'
    # data_dir = 'D:/graduate/G2/xinke/SEEDV/EEG_processed_sxk'
    data_dir2 = '/mnt/data/model_weights/grm/SEEDV/EEG_processed_sampled'
    # data_dir2 = 'D:/graduate/G2/xinke/SEEDV/EEG_processed_sampled'
    timeLen = 5
    timeStep = 2
    fs = 250
    n_channs = 62
    n_session = 3

    data, onesub_labels, n_samples_onesub, n_samples_sessions = load_processed_SEEDV_data(data_dir,fs,n_channs,timeLen,timeStep,n_session)
    print(data.shape)
    print(onesub_labels)
    print(n_samples_onesub)
    print(n_samples_sessions)
    sampled_data = {}
    sampled_data['data'] = data
    sampled_data['onesub_labels'] = onesub_labels
    sampled_data['n_samples_onesub'] = n_samples_onesub
    sampled_data['n_samples_sessions'] = n_samples_sessions

def test_load_processed_SEEDV_NEW_data():
    data_dir = '/mnt/data/model_weights/grm/SEEDV-NEW/processed_data'
    # data_dir = '/mnt/data/model_weights/grm/SEEDV_new2/processed_ddata'
    # data_dir = 'D:/graduate/G2/xinke/SEEDV/EEG_processed_sxk'
    data_dir2 = '/mnt/data/model_weights/grm/SEEDV/EEG_processed_sampled'
    # data_dir2 = 'D:/graduate/G2/xinke/SEEDV/EEG_processed_sampled'
    timeLen = 5
    timeStep = 2
    fs = 125
    n_channs = 60
    n_session = 3

    data, onesub_labels, n_samples_onesub, n_samples_sessions = load_processed_SEEDV_NEW_data(data_dir,fs,n_channs,timeLen,timeStep,n_session)
    print(data.shape)
    print(onesub_labels)
    print(n_samples_onesub)
    print(n_samples_sessions)
    sampled_data = {}
    sampled_data['data'] = data
    sampled_data['onesub_labels'] = onesub_labels
    sampled_data['n_samples_onesub'] = n_samples_onesub
    sampled_data['n_samples_sessions'] = n_samples_sessions

if __name__ == '__main__':
    # test_load_processed_SEEDV_data()
    test_load_processed_SEEDV_NEW_data()