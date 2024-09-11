import scipy.io as sio
from glob import glob
import hdf5storage
import numpy as np
import os
import copy


def video_order_load(n_vids=28):
    datapath = '/home/gpt/grm/CLISA_Hydra/After_remarks'
    filesPath = os.listdir(datapath)
    filesPath.sort()
    vid_orders = np.zeros((len(filesPath), n_vids),dtype=int)
    for idx, file in enumerate(filesPath):
        # Here don't forget to arange the subjects arrangement
        # print(file)
        remark_file = os.path.join(datapath,file,'After_remarks.mat')
        subject_remark = hdf5storage.loadmat(remark_file)['After_remark']
        vid_orders[idx, :] = [np.squeeze(subject_remark[vid][0][2]) for vid in range(0,n_vids)]
    # print('vid_order shape: ', vid_orders.shape)
    return vid_orders


def reorder_vids(data, n_vids, vid_play_order):
    # data: (n_subs, n_points, n_feas)
    # return:(n_subs, n_points, n_feas)
    n_subs = data.shape[0]
    n_samples = data.shape[1]//n_vids
    vid_play_order_copy = vid_play_order.copy()
    if n_vids == 24:
        vid_play_order_new = np.zeros((n_subs, n_vids)).astype(np.int32)
        data_reorder = np.zeros_like(data)
        for sub in range(n_subs):
            tmp = vid_play_order_copy[sub,:]
            tmp = tmp[(tmp<13)|(tmp>16)]
            tmp[tmp>=17] = tmp[tmp>=17] - 4
            tmp = tmp - 1
            vid_play_order_new[sub, :] = tmp

            data_sub = data[sub, :, :]
            data_sub = data_sub.reshape(n_vids, n_samples, data_sub.shape[-1])
            data_sub = data_sub[tmp, :, :]
            data_reorder[sub, :, :] = data_sub.reshape(n_vids*n_samples, data_sub.shape[-1])
    elif n_vids == 28:
        vid_play_order_new = np.zeros((n_subs, n_vids)).astype(np.int32)
        data_reorder = np.zeros_like(data)
        for sub in range(n_subs):
            tmp = vid_play_order_copy[sub,:]
            tmp = tmp - 1
            vid_play_order_new[sub, :] = tmp

            data_sub = data[sub, :, :]
            data_sub = data_sub.reshape(n_vids, n_samples, data_sub.shape[-1])
            # Error occurs saying that the elements of tmp is not int
            tmp = [int(i) for i in tmp]
            data_sub = data_sub[tmp, :, :]
            data_reorder[sub, :, :] = data_sub.reshape(n_vids*n_samples, data_sub.shape[-1])
    return data_reorder, vid_play_order_new

def reorder_vids_sepVideo(data, vid_play_order, sel_vid_inds, n_vids_all):
    # data: (n_subs, n_points, n_feas)
    n_vids = len(sel_vid_inds)
    n_subs = data.shape[0]
    # print('n_subs:',n_subs)
    vid_play_order_copy = vid_play_order.copy()
    vid_play_order_new = np.zeros((n_subs, len(sel_vid_inds))).astype(np.int32)
    data_reorder = np.zeros_like(data)
    if n_vids_all == 24:
        for sub in range(n_subs):
            tmp = vid_play_order_copy[sub,:]
            tmp = tmp[(tmp<13)|(tmp>16)]
            tmp[tmp>=17] = tmp[tmp>=17] - 4
            tmp = tmp - 1

            tmp_new = []
            for i in range(len(tmp)):
                if tmp[i] in sel_vid_inds:
                    tmp_new.append(np.where(sel_vid_inds==tmp[i])[0][0])
            tmp_new = np.array(tmp_new)

            vid_play_order_new[sub, :] = tmp_new

            data_sub = data[sub, :, :]
            data_sub = data_sub.reshape(n_vids, -1, data_sub.shape[-1])
            data_sub = data_sub[tmp_new, :, :]
            data_reorder[sub, :, :] = data_sub.reshape(-1, data_sub.shape[-1])
    elif n_vids_all == 28:
        for sub in range(n_subs):
            tmp = vid_play_order_copy[sub,:]
            tmp = tmp - 1

            tmp_new = []
            for i in range(len(tmp)):
                if tmp[i] in sel_vid_inds:
                    tmp_new.append(np.where(sel_vid_inds==tmp[i])[0][0])
            tmp_new = np.array(tmp_new)

            vid_play_order_new[sub, :] = tmp_new

            data_sub = data[sub, :, :]
            data_sub = data_sub.reshape(n_vids, -1, data_sub.shape[-1])
            data_sub = data_sub[tmp_new, :, :]
            data_reorder[sub, :, :] = data_sub.reshape(-1, data_sub.shape[-1])
        
    return data_reorder, vid_play_order_new



def reorder_vids_back(data, n_vids, vid_play_order_new):
    # data: (n_subs, n_points, n_feas)
    # return:(n_subs, n_points, n_feas)
    n_subs = data.shape[0]
    n_samples = data.shape[1]//n_vids


    data_back = np.zeros((n_subs, n_vids, n_samples, data.shape[-1]))

    for sub in range(n_subs):
        data_sub = data[sub, :, :].reshape(n_vids, n_samples, data.shape[-1])
        data_back[sub, vid_play_order_new[sub, :], :, :] = data_sub
    data_back = data_back.reshape(n_subs, n_vids*n_samples, data.shape[-1])
    return data_back


if __name__ == '__main__':
    video_order_load(28)
