import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from transformers.models.bert import BertModel, BertConfig


def stratified_layerNorm(out, n_samples):
    n_subs = int(out.shape[0] / n_samples)
    out_str = out.clone()
    for i in range(n_subs):
        out_oneSub = out[n_samples*i: n_samples*(i+1)]
        out_oneSub = out_oneSub.reshape(out_oneSub.shape[0], -1, out_oneSub.shape[-1]).permute(0,2,1)
        out_oneSub = out_oneSub.reshape(out_oneSub.shape[0]*out_oneSub.shape[1], -1)
        # out_oneSub[torch.isinf(out_oneSub)] = -50
        # out_oneSub[torch.isnan(out_oneSub)] = -50
        out_oneSub_str = out_oneSub.clone()
        # We don't care about the channels with very small activations
        # out_oneSub_str[:, out_oneSub.abs().sum(dim=0) > 1e-4] = (out_oneSub[:, out_oneSub.abs().sum(dim=0) > 1e-4] - out_oneSub[
        #     :, out_oneSub.abs().sum(dim=0) > 1e-4].mean(dim=0)) / (out_oneSub[:, out_oneSub.abs().sum(dim=0) > 1e-4].std(dim=0) + 1e-3)
        out_oneSub_str = (out_oneSub - out_oneSub.mean(dim=0)) / (out_oneSub.std(dim=0) + 1e-3)
        out_str[n_samples*i: n_samples*(i+1)] = out_oneSub_str.reshape(n_samples, -1, out_oneSub_str.shape[1]).permute(0,2,1).reshape(n_samples, out.shape[1], out.shape[2], -1)
        # out_str[torch.isnan(out_str)]=1
    return out_str

# def stratified_layerNorm_test(out, n_samples):
#     n_subs = int(out.shape[0] / n_samples)
#     out_str = out.clone()
#     for i in range(n_subs):
#         out_oneSub = out[n_samples*i: n_samples*(i+1)]
#         out_oneSub = out_oneSub.reshape(out_oneSub.shape[0], -1, out_oneSub.shape[-1]).permute(0,2,1)
#         out_oneSub = out_oneSub.reshape(out_oneSub.shape[0]*out_oneSub.shape[1], -1)
#         out_oneSub_str = out_oneSub.clone()
#         # We don't care about the channels with very small activations
#         # out_oneSub_str[:, out_oneSub.abs().sum(dim=0) > 1e-4] = (out_oneSub[:, out_oneSub.abs().sum(dim=0) > 1e-4] - out_oneSub[
#         #     :, out_oneSub.abs().sum(dim=0) > 1e-4].mean(dim=0)) / (out_oneSub[:, out_oneSub.abs().sum(dim=0) > 1e-4].std(dim=0) + 1e-3)
#         out_oneSub_str = (out_oneSub - out_oneSub.mean(dim=0)) / (out_oneSub.std(dim=0) + 1e-3)
#         # print(out_oneSub.mean(dim=0),out_oneSub.std(dim=0))
#         a = torch.argmin(out_oneSub.mean(dim=0))
#         b = torch.argmin(out_oneSub.std(dim=0))
#         c = torch.argmax(out_oneSub.std(dim=0))
        
#         print(a)
#         print(b,c)
#         print(min(out_oneSub.std(dim=0)),max(out_oneSub.std(dim=0)))
#         print(out_oneSub[:,int(a)].numpy().tolist)
#         print(out_oneSub[:,int(b)].numpy().tolist)
#         print(out_oneSub[:,int(c)].numpy().tolist)

#         out_str[n_samples*i: n_samples*(i+1)] = out_oneSub_str.reshape(n_samples, -1, out_oneSub_str.shape[1]).permute(0,2,1).reshape(n_samples, out.shape[1], out.shape[2], -1)
#     return out_str


class ConvNet_complete_baseline(nn.Module):
    def __init__(self, n_timeFilters, timeFilterLen, n_spatialFilters, avgPoolLen, timeSmootherLen, n_channs, stratified, multiFact, saveFea):
        super(ConvNet_complete_baseline, self).__init__()
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, (timeFilterLen-1)//2))
        self.spatialConv = nn.Conv2d(n_timeFilters, n_timeFilters*n_spatialFilters, (n_channs, 1), groups=n_timeFilters)
        self.avgpool = nn.AvgPool2d((1, avgPoolLen))
        n_msFilters_total = n_timeFilters*n_spatialFilters
        self.timeConv1 = nn.Conv2d(n_msFilters_total, n_msFilters_total * multiFact, (1, timeSmootherLen), groups=n_msFilters_total)
        self.timeConv2 = nn.Conv2d(n_msFilters_total * multiFact, n_msFilters_total * multiFact * multiFact, (1, timeSmootherLen), groups=n_msFilters_total * multiFact)
        self.stratified = stratified
        self.saveFea = saveFea

    def forward(self, input):
        if 'initial' in self.stratified:
            input = stratified_layerNorm(input, int(input.shape[0]/2))
        out = self.timeConv(input)
        out = F.relu(self.spatialConv(out))
        out = self.avgpool(out)

        if self.saveFea:
            out = out.reshape(out.shape[0], -1)
            return out
        else:
            if 'middle1' in self.stratified:
                out = stratified_layerNorm(out, int(out.shape[0]/2))
            out = F.relu(self.timeConv1(out))
            out = F.relu(self.timeConv2(out))
            if 'middle2' in self.stratified:
                out = stratified_layerNorm(out, int(out.shape[0]/2))
            out = out.reshape(out.shape[0], -1)
            return out

class ConvNet_complete_baseline_new(nn.Module):
    def __init__(self, n_timeFilters, timeFilterLen, n_spatialFilters, stimeFilterLen, avgPoolLen, timeSmootherLen, n_channs, stratified, multiFact, saveFea, extract_mode):
        super(ConvNet_complete_baseline_new, self).__init__()
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, (timeFilterLen-1)//2))
        self.spatialConv = nn.Conv2d(n_timeFilters, n_timeFilters*n_spatialFilters, (n_channs, stimeFilterLen), groups=n_timeFilters)
        self.avgpool = nn.AvgPool2d((1, avgPoolLen))
        n_msFilters_total = n_timeFilters*n_spatialFilters
        self.timeConv1 = nn.Conv2d(n_msFilters_total, n_msFilters_total * multiFact, (1, timeSmootherLen), groups=n_msFilters_total)
        self.timeConv2 = nn.Conv2d(n_msFilters_total * multiFact, n_msFilters_total * multiFact * multiFact, (1, timeSmootherLen), groups=n_msFilters_total * multiFact)
        self.stratified = stratified
        self.saveFea = saveFea
        self.stimeFilterLen = stimeFilterLen
        self.extract_mode = extract_mode

    def forward(self, input):
        if 'initial' in self.stratified:
            input = stratified_layerNorm(input, int(input.shape[0]/2))

        out = self.timeConv(input)

        p = (self.stimeFilterLen - 1)
        out = self.spatialConv(F.pad(out, (int(p//2), p-int(p//2)), "constant", 0))

        fea_out = out.clone()
        if self.extract_mode == 'me':
            fea_out = F.elu(fea_out)
            fea_out = self.avgpool(fea_out)
        
        

        if self.saveFea:
            # out = out.reshape(out.shape[0], -1)

            return fea_out
        else:
            out = F.elu(out)
            out = self.avgpool(out)
            if 'middle1' in self.stratified:
                out = stratified_layerNorm(out, int(out.shape[0]/2))
            out = F.relu(self.timeConv1(out))
            out = F.relu(self.timeConv2(out))
            if 'middle2' in self.stratified:
                out = stratified_layerNorm(out, int(out.shape[0]/2))
            proj_out = out.reshape(out.shape[0], -1)
            return F.normalize(proj_out, dim=1)
    
    def set_saveFea(self, saveFea):
        self.saveFea = saveFea

    def set_extract_mode(self, extract_mode):
        self.extract_mode = extract_mode

    def set_stratified(self,stratified):
        self.stratified = stratified


class ConvNet_attention_simple(nn.Module):
    def __init__(self, n_timeFilters, timeFilterLen0, n_msFilters, timeFilterLen, avgPoolLen, timeSmootherLen, n_channs, stratified, multiFact, activ, temp, saveFea):
        super(ConvNet_attention_simple, self).__init__()
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen0), padding=(0, (timeFilterLen0-1)//2))
        self.msConv1 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, timeFilterLen), groups=n_timeFilters)
        self.msConv2 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, timeFilterLen), dilation=(1,3), groups=n_timeFilters)
        self.msConv3 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, timeFilterLen), dilation=(1,6), groups=n_timeFilters)
        self.msConv4 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, timeFilterLen), dilation=(1,12), groups=n_timeFilters)

        n_msFilters_total = n_timeFilters * n_msFilters * 4

        # Attention
        seg_att = 15
        self.att_conv = nn.Conv2d(n_msFilters_total, n_msFilters_total, (1, seg_att), groups=n_msFilters_total)
        self.att_pool = nn.AvgPool2d((1, seg_att), stride=1)
        self.att_pointConv = nn.Conv2d(n_msFilters_total, n_msFilters_total, (1, 1))

        self.avgpool = nn.AvgPool2d((1, avgPoolLen))
        n_msFilters_total = n_timeFilters * n_msFilters * 4
        self.timeConv1 = nn.Conv2d(n_msFilters_total, n_msFilters_total * multiFact, (1, timeSmootherLen), groups=n_msFilters_total)
        self.timeConv2 = nn.Conv2d(n_msFilters_total * multiFact, n_msFilters_total * multiFact * multiFact, (1, timeSmootherLen), groups=n_msFilters_total * multiFact)
        self.stratified = stratified
        self.timeFilterLen = timeFilterLen
        self.saveFea = saveFea
        self.activ = activ
        self.temp = temp

    def forward(self, input):
        if 'initial' in self.stratified:
            input = stratified_layerNorm(input, int(input.shape[0]/2))
        out = self.timeConv(input)
        p = np.array([1,3,6,12]) * (self.timeFilterLen - 1)
        out1 = self.msConv1(F.pad(out, (int(p[0]//2), p[0]-int(p[0]//2)), "constant", 0))
        out2 = self.msConv2(F.pad(out, (int(p[1]//2), p[1]-int(p[1]//2)), "constant", 0))
        out3 = self.msConv3(F.pad(out, (int(p[2]//2), p[2]-int(p[2]//2)), "constant", 0))
        out4 = self.msConv4(F.pad(out, (int(p[3]//2), p[3]-int(p[3]//2)), "constant", 0))
        # print(out1.shape, out2.shape, out3.shape, out4.shape)
        out = torch.cat((out1, out2, out3, out4), 1) # (B, dims, 1, T)

        # Attention
        att_w = F.relu(self.att_conv(F.pad(out, (14, 0), "constant", 0)))
        att_w = self.att_pool(F.pad(att_w, (14, 0), "constant", 0)) # (B, dims, 1, T)
        att_w = self.att_pointConv(att_w)
        if self.activ == 'relu':
            att_w = F.relu(att_w)
        elif self.activ == 'softmax':
            att_w = F.softmax(att_w / self.temp, dim=1)
        out = att_w * F.relu(out)

        if self.saveFea:
            return out
        else:
            out = self.avgpool(out)
            if 'middle1' in self.stratified:
                out = stratified_layerNorm(out, int(out.shape[0]/2))
            out = F.relu(self.timeConv1(out))
            out = F.relu(self.timeConv2(out))
            if 'middle2' in self.stratified:
                out = stratified_layerNorm(out, int(out.shape[0]/2))
            out = out.reshape(out.shape[0], -1)
            return out

class simpleNN3(nn.Module):
    def __init__(self, inp_dim, hidden_dim=[128,64], out_dim=9, dropout=0.2, bn='no'):
        super(simpleNN3, self).__init__()
        self.fc1 = nn.Linear(inp_dim, hidden_dim[0])
        # if (bn == 'bn1') or (bn == 'bn2'):
        self.bn1 = nn.BatchNorm1d(hidden_dim[0], affine=False)
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        # if bn == 'bn2':
        self.bn2 = nn.BatchNorm1d(hidden_dim[1], affine=False)
        self.fc3 = nn.Linear(hidden_dim[1], out_dim)
        self.bn = bn
        self.drop = nn.Dropout(p=dropout)
        # self.flag = False
    def forward(self, input):

        out = F.relu(self.fc1(input))

        if (self.bn == 'bn1') or (self.bn == 'bn2'):
            out = self.bn1(out)
        out = self.drop(out)
        out = F.relu(self.fc2(out))

        if self.bn == 'bn2':
            out = self.bn2(out)
        out = self.drop(out)
        out = self.fc3(out)

        return out
    
    # def set_debug_flag(self,flag):
    #     self.flag = flag

class simpleNN3_seed(nn.Module):
    def __init__(self, inp_dim, hidden_dim, out_dim):
        super(simpleNN3, self).__init__()
        self.fc1 = nn.Linear(inp_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
    def forward(self, input):
        out = F.relu(self.fc1(input))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class Conv_att_simple_new(nn.Module):
    # 配置说明 125Hz采样率基线  使用参数  dilation_array=[1,3,6,12]      seg_att = 15  avgPoolLen = 15  timeSmootherLen=3 mslen = 2,3   如果频率变化请在基线上乘以相应倍数
    def __init__(self, n_timeFilters, timeFilterLen, n_msFilters, msFilter_timeLen, n_channs=64, dilation_array=np.array([1,6,12,24]), seg_att=30, avgPoolLen = 30,
                  timeSmootherLen=6, multiFact=2, stratified=[], activ='softmax', temp=1.0, saveFea=True, has_att=True, extract_mode='me', global_att=False):
        super().__init__()
        self.stratified = stratified
        self.msFilter_timeLen = msFilter_timeLen
        self.activ = activ
        self.temp = temp
        self.dilation_array = np.array(dilation_array)   
        self.saveFea = saveFea
        self.has_att = has_att
        self.extract_mode = extract_mode
        self.global_att = global_att
        

        # time and spacial conv
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, (timeFilterLen-1)//2))
        self.msConv1 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, msFilter_timeLen), groups=n_timeFilters)
        self.msConv2 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, msFilter_timeLen), dilation=(1,self.dilation_array[1]), groups=n_timeFilters)
        self.msConv3 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, msFilter_timeLen), dilation=(1,self.dilation_array[2]), groups=n_timeFilters)
        self.msConv4 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, msFilter_timeLen), dilation=(1,self.dilation_array[3]), groups=n_timeFilters)

        n_msFilters_total = n_timeFilters * n_msFilters * 4

        # Attention
        self.seg_att = seg_att               #  *2 等比缩放
        self.att_conv = nn.Conv2d(n_msFilters_total, n_msFilters_total, (1, self.seg_att), groups=n_msFilters_total)
        self.att_pool = nn.AvgPool2d((1, self.seg_att), stride=1)
        self.att_pointConv = nn.Conv2d(n_msFilters_total, n_msFilters_total, (1, 1))

        # projector avepooling+timeSmooth
        self.avgpool = nn.AvgPool2d((1, avgPoolLen))
        self.timeConv1 = nn.Conv2d(n_msFilters_total, n_msFilters_total * multiFact, (1, timeSmootherLen), groups=n_msFilters_total)
        self.timeConv2 = nn.Conv2d(n_msFilters_total * multiFact, n_msFilters_total * multiFact * multiFact, (1, timeSmootherLen), groups=n_msFilters_total * multiFact)
        # # pooling  时间上的max pooling目前不需要，因为最后输出层特征会整体做个时间上的平均,时间上用ave比max更符合直觉
        # self.maxPoolLen = maxPoolLen
        # self.maxpool = nn.MaxPool2d((1, self.maxPoolLen),self.maxPoolLen)
        # # self.flatten = nn.Flatten()
    
    def forward(self, input):
        if 'initial' in self.stratified:
            input = stratified_layerNorm(input, int(input.shape[0]/2))
        out = self.timeConv(input)
        p = self.dilation_array * (self.msFilter_timeLen - 1)
        out1 = self.msConv1(F.pad(out, (int(p[0]//2), p[0]-int(p[0]//2)), "constant", 0))
        out2 = self.msConv2(F.pad(out, (int(p[1]//2), p[1]-int(p[1]//2)), "constant", 0))
        out3 = self.msConv3(F.pad(out, (int(p[2]//2), p[2]-int(p[2]//2)), "constant", 0))
        out4 = self.msConv4(F.pad(out, (int(p[3]//2), p[3]-int(p[3]//2)), "constant", 0))
        # print(out1.shape, out2.shape, out3.shape, out4.shape)
        out = torch.cat((out1, out2, out3, out4), 1) # (B, dims, 1, T)

        # Attention
        if self.has_att:
            att_w = F.relu(self.att_conv(F.pad(out, (self.seg_att-1, 0), "constant", 0)))
            if self.global_att:
                att_w = torch.mean(F.pad(att_w, (self.seg_att-1, 0), "constant", 0),-1).unsqueeze(-1) # (B, dims, 1, 1)
            else:
                att_w = self.att_pool(F.pad(att_w, (self.seg_att-1, 0), "constant", 0)) # (B, dims, 1, T)
            att_w = self.att_pointConv(att_w)
            if self.activ == 'relu':
                att_w = F.relu(att_w)
            elif self.activ == 'softmax':
                att_w = F.softmax(att_w / self.temp, dim=1)
            elif self.activ == 'sigmoid':
                att_w = F.sigmoid(att_w)
            out = att_w * F.relu(out)          # (B, dims, 1, T)
        else:
            if self.extract_mode == 'me':
                out = F.relu(out)


        if self.saveFea:
            return out
        else:         # projecter
            if self.extract_mode == 'de':
                out = F.relu(out)
            out = self.avgpool(out)    # B*(t_dim*n_msFilters*4)*1*t_pool
            if 'middle1' in self.stratified:
                out = stratified_layerNorm(out, int(out.shape[0]/2))
            out = F.relu(self.timeConv1(out))
            out = F.relu(self.timeConv2(out))          #B*(t_dim*n_msFilters*4*multiFact*multiFact)*1*t_pool
            if 'middle2' in self.stratified:
                out = stratified_layerNorm(out, int(out.shape[0]/2))     
            proj_out = out.reshape(out.shape[0], -1)
            return F.normalize(proj_out, dim=1)

    
    def set_saveFea(self, saveFea):
        self.saveFea = saveFea

    def set_stratified(self,stratified):
        self.stratified = stratified

class Conv_att_timefilter(nn.Module):
    # 配置说明 125Hz采样率基线  使用参数  dilation_array=[1,3,6,12]      seg_att = 15  avgPoolLen = 15  timeSmootherLen=3 mslen = 2,3   如果频率变化请在基线上乘以相应倍数
    def __init__(self, n_timeFilters, timeFilterLen, n_msFilters, msFilter_timeLen, n_channs=64, dilation_array=np.array([1,6,12,24]), seg_att=30, avgPoolLen = 30,
                  timeSmootherLen=6, multiFact=2, stratified=[], activ='softmax', temp=1.0, saveFea=True, has_att=True, extract_mode='me', global_att=False):
        super().__init__()
        self.stratified = stratified
        self.msFilter_timeLen = msFilter_timeLen
        self.activ = activ
        self.temp = temp
        self.dilation_array = dilation_array   
        self.saveFea = saveFea
        self.has_att = has_att
        self.extract_mode = extract_mode
        self.global_att = global_att
        

        # time and spacial conv
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, (timeFilterLen-1)//2))
        self.msConv1 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, msFilter_timeLen), groups=n_timeFilters)
        self.msConv2 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, msFilter_timeLen), dilation=(1,self.dilation_array[1]), groups=n_timeFilters)
        self.msConv3 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, msFilter_timeLen), dilation=(1,self.dilation_array[2]), groups=n_timeFilters)
        self.msConv4 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, msFilter_timeLen), dilation=(1,self.dilation_array[3]), groups=n_timeFilters)

        n_msFilters_total = n_timeFilters * n_msFilters * 4

        # Attention
        self.seg_att = seg_att               #  *2 等比缩放
        self.att_conv = nn.Conv2d(n_msFilters_total, n_msFilters_total, (1, self.seg_att), groups=n_msFilters_total)
        self.att_pool = nn.AvgPool2d((1, self.seg_att), stride=1)
        self.att_pointConv = nn.Conv2d(n_msFilters_total, n_msFilters_total, (1, 1))

        # projector avepooling+timeSmooth
        self.avgpool = nn.AvgPool2d((1, avgPoolLen))
        self.timeConv1 = nn.Conv2d(n_msFilters_total, n_msFilters_total * multiFact, (1, timeSmootherLen), groups=n_msFilters_total)
        self.timeConv2 = nn.Conv2d(n_msFilters_total * multiFact, n_msFilters_total * multiFact * multiFact, (1, timeSmootherLen), groups=n_msFilters_total * multiFact)
        # # pooling  时间上的max pooling目前不需要，因为最后输出层特征会整体做个时间上的平均,时间上用ave比max更符合直觉
        # self.maxPoolLen = maxPoolLen
        # self.maxpool = nn.MaxPool2d((1, self.maxPoolLen),self.maxPoolLen)
        # # self.flatten = nn.Flatten()
    
    def forward(self, input):

        out = self.timeConv(input)
        
        return out
       

class Conv_att_transformer(nn.Module):
    # 配置说明 125Hz采样率基线  使用参数  dilation_array=[1,3,6,12]      seg_att = 15  avgPoolLen = 15  timeSmootherLen=3 mslen = 2,3   如果频率变化请在基线上乘以相应倍数
    def __init__(self, n_timeFilters, timeFilterLen, n_msFilters, msFilter_timeLen, n_channs=64, dilation_array=np.array([1,6,12,24]), avgPoolLen = 30,
                  timeSmootherLen=6, multiFact=2, stratified=[], activ='softmax', temp=1.0, saveFea=True, has_att=True, extract_mode='me', attention_mode='channel'):
        super().__init__()
        self.stratified = stratified
        self.msFilter_timeLen = msFilter_timeLen
        self.activ = activ
        self.temp = temp
        self.dilation_array = np.array(dilation_array)   
        self.saveFea = saveFea
        self.has_att = has_att
        self.extract_mode = extract_mode
        self.attention_mode = attention_mode
        

        # time and spacial conv
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, (timeFilterLen-1)//2))
        self.msConv1 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, msFilter_timeLen), groups=n_timeFilters)
        self.msConv2 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, msFilter_timeLen), dilation=(1,self.dilation_array[1]), groups=n_timeFilters)
        self.msConv3 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, msFilter_timeLen), dilation=(1,self.dilation_array[2]), groups=n_timeFilters)
        self.msConv4 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, msFilter_timeLen), dilation=(1,self.dilation_array[3]), groups=n_timeFilters)

        n_msFilters_total = n_timeFilters * n_msFilters * 4

        num_head = 4
        num_hidden_layers = 1 
        # transformer_layer
        if attention_mode == 'channel':
            em_dim = 200
            max_len = n_msFilters_total
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, em_dim))
        elif attention_mode == 'time':
            em_dim = n_msFilters_total
            max_len = 5*125
            
        transfomer_config = BertConfig(vocab_size=1, hidden_size=em_dim,
                                            num_hidden_layers=num_hidden_layers, num_attention_heads=num_head,
                                            intermediate_size=int(2 * em_dim),
                                            hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                                            max_position_embeddings=max_len,
                                            is_decoder=True)   # todo: magic param, 3layers?
        self.transformer_layer = BertModel(transfomer_config)   

        # projector avepooling+timeSmooth
        self.avgpool = nn.AvgPool2d((1, avgPoolLen))
        self.timeConv1 = nn.Conv2d(n_msFilters_total, n_msFilters_total * multiFact, (1, timeSmootherLen), groups=n_msFilters_total)
        self.timeConv2 = nn.Conv2d(n_msFilters_total * multiFact, n_msFilters_total * multiFact * multiFact, (1, timeSmootherLen), groups=n_msFilters_total * multiFact)
        # # pooling  时间上的max pooling目前不需要，因为最后输出层特征会整体做个时间上的平均,时间上用ave比max更符合直觉
        # self.maxPoolLen = maxPoolLen
        # self.maxpool = nn.MaxPool2d((1, self.maxPoolLen),self.maxPoolLen)
        # # self.flatten = nn.Flatten()
    
    def forward(self, input):
        if 'initial' in self.stratified:
            input = stratified_layerNorm(input, int(input.shape[0]/2))
        out = self.timeConv(input)
        p = self.dilation_array * (self.msFilter_timeLen - 1)
        out1 = self.msConv1(F.pad(out, (int(p[0]//2), p[0]-int(p[0]//2)), "constant", 0))
        out2 = self.msConv2(F.pad(out, (int(p[1]//2), p[1]-int(p[1]//2)), "constant", 0))
        out3 = self.msConv3(F.pad(out, (int(p[2]//2), p[2]-int(p[2]//2)), "constant", 0))
        out4 = self.msConv4(F.pad(out, (int(p[3]//2), p[3]-int(p[3]//2)), "constant", 0))
        # print(out1.shape, out2.shape, out3.shape, out4.shape)
        out = torch.cat((out1, out2, out3, out4), 1) # (B, dims, 1, T)
        bs, dims, _, seq_len = out.shape
        # Attention
        if self.has_att:
            # att_w = F.relu(self.att_conv(F.pad(out, (self.seg_att-1, 0), "constant", 0)))
            # if self.global_att:
            #     att_w = torch.mean(F.pad(att_w, (self.seg_att-1, 0), "constant", 0),-1).unsqueeze(-1) # (B, dims, 1, 1)
            # else:
            #     att_w = self.att_pool(F.pad(att_w, (self.seg_att-1, 0), "constant", 0)) # (B, dims, 1, T)
            # att_w = self.att_pointConv(att_w)
            # if self.activ == 'relu':
            #     att_w = F.relu(att_w)
            # elif self.activ == 'softmax':
            #     att_w = F.softmax(att_w / self.temp, dim=1)
            # out = att_w * F.relu(out)          # (B, dims, 1, T)
            if self.attention_mode == 'channel':
                out = self.adaptive_pool(out)
                out = out.reshape(bs, dims, -1)
                out, _ = self.transformer_layer(inputs_embeds=out, 
                                       attention_mask=torch.ones(bs, dims, device=out.device, dtype=torch.long),
                                        position_ids=torch.ones(bs, dims, device=out.device, dtype=torch.long),
                                        return_dict=False)
                # print(len(out))
                out = out.unsqueeze(2)
                # print(out.shape)
            elif self.attention_mode == 'time':
                out = out.squeeze().transpose(1,2)
                # print(out.shape)
                position_ids = torch.arange(seq_len, device=out.device, dtype=torch.long).unsqueeze(0).expand(bs, seq_len) 
                out = self.transformer_layer(inputs_embeds=out, 
                                       attention_mask=torch.ones(bs, seq_len, device=out.device, dtype=torch.long),
                                        position_ids=position_ids,
                                        return_dict=False)
                out = out[0]
                out = out.transpose(1,2).unsqueeze(2)
                # print(out.shape)

        else:
            if self.extract_mode == 'me':
                out = F.relu(out)


        if self.saveFea:
            return out
        else:         # projecter
            if self.extract_mode == 'de':
                out = F.relu(out)
            out = self.avgpool(out)    # B*(t_dim*n_msFilters*4)*1*t_pool
            if 'middle1' in self.stratified:
                out = stratified_layerNorm(out, int(out.shape[0]/2))
            out = F.relu(self.timeConv1(out))
            out = F.relu(self.timeConv2(out))          #B*(t_dim*n_msFilters*4*multiFact*multiFact)*1*t_pool
            if 'middle2' in self.stratified:
                out = stratified_layerNorm(out, int(out.shape[0]/2))     
            proj_out = out.reshape(out.shape[0], -1)
            return F.normalize(proj_out, dim=1)

    
    def set_saveFea(self, saveFea):
        self.saveFea = saveFea

    def set_stratified(self,stratified):
        self.stratified = stratified
