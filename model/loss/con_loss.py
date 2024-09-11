import torch
# import torch.nn.functional as F
import torch.nn as nn
# import numpy as np
# import time

# class SimCLR(object):

#     def __init__(self, args, model, optimizer, scheduler, log_dir, stratified):
#         self.args = args
#         self.model = model.to(args.device)
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
#         self.stratified = stratified
#         self.log_dir = log_dir

#     def info_nce_loss(self, features, stratified):
#         # print(features.shape)
#         bs = int(features.shape[0] // 2)
#         labels = torch.cat([torch.arange(bs) for i in range(2)], dim=0)
#         labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
#         labels = labels.to(self.args.device)

#         # Normlize the features according to subject
#         if stratified == 'stratified':
#             features_str = features.clone()
#             features_str[:bs, :] = (features[:bs, :] -  features[:bs, :].mean(
#                 dim=0)) / (features[:bs, :].std(dim=0) + 1e-3)
#             features_str[bs:, :] = (features[bs:, :] -  features[bs:, :].mean(
#                 dim=0)) / (features[bs:, :].std(dim=0) + 1e-3)
#             features = F.normalize(features_str, dim=1)
#         elif stratified == 'bn':
#             features_str = features.clone()
#             features_str = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-3)
#             features = F.normalize(features_str, dim=1)
#         elif stratified == 'no':
#             features = F.normalize(features, dim=1)

#         similarity_matrix = torch.matmul(features, features.T)

#         # discard the main diagonal from both: labels and similarities matrix
#         mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
#         labels = labels[~mask].view(labels.shape[0], -1)
#         similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

#         # select and combine multiple positives
#         positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

#         # select only the negatives
#         negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

#         # Put the positive column at the end (when all the entries are the same, the top1 acc will be 0; while if the
#         # positive column is at the start, the top1 acc might be exaggerated)
#         logits = torch.cat([negatives, positives], dim=1)
#         # The label means the last column contain the positive pairs
#         labels = torch.ones(logits.shape[0], dtype=torch.long)*(logits.shape[1]-1)
#         labels = labels.to(self.args.device)

#         logits = logits / self.args.temperature
#         return logits, labels


#     def train(self, train_loader, val_loader):
#         n_iter = 0

#         bad_count = 0
#         best_acc, best_loss = -1, 1000
#         model_epochs, optimizer_epochs = {}, {}
#         train_top1_history, val_top1_history = np.zeros(self.args.epochs_pretrain), np.zeros(self.args.epochs_pretrain)
#         train_top5_history, val_top5_history = np.zeros(self.args.epochs_pretrain), np.zeros(self.args.epochs_pretrain)
#         train_loss_history, val_loss_history = np.zeros(self.args.epochs_pretrain), np.zeros(self.args.epochs_pretrain)
#         for epoch_counter in range(self.args.epochs_pretrain):
#             start_time = time.time()
#             train_loss = 0
#             train_acc = 0
#             train_acc5 = 0
#             # train_loss_mediate, train_acc_mediate = 0, 0
#             # loss_batch = torch.tensor(0., requires_grad=True)
#             self.model.train()
#             for count, (data, labels) in enumerate(train_loader):
#                 data = data.to(self.args.device)
#                 # with autocast(enabled=self.args.fp16_precision):
#                 features = self.model(data)
#                 # print(self.model.msConv.weight.data[:3])
#                 logits, labels = self.info_nce_loss(features, self.stratified)
#                 loss = self.criterion(logits, labels)
#                 # loss_batch = loss_batch + loss

#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()

#                 top1, top5 = accuracy(logits, labels, topk=(1,5))

#                 n_iter += 1

#                 train_loss = train_loss + loss.data.cpu().numpy()
#                 train_acc = train_acc + top1[0].cpu().numpy()
#                 train_acc5 = train_acc5 + top5[0].cpu().numpy()

#             train_loss = train_loss / (count + 1)
#             train_acc = train_acc / (count + 1)
#             train_acc5 = train_acc5 / (count + 1)

#             val_loss = 0
#             val_acc = 0
#             val_acc5 = 0
#             self.model.eval()
#             for count, (data, labels) in enumerate(val_loader):
#                 data = data.to(self.args.device)

#                 features = self.model(data)
#                 logits, labels = self.info_nce_loss(features, self.stratified)
#                 loss = self.criterion(logits, labels)

#                 top1, top5 = accuracy(logits, labels, topk=(1,5))

#                 val_loss = val_loss + loss.data.cpu().numpy()
#                 val_acc = val_acc + top1[0].cpu().numpy()
#                 val_acc5 = val_acc5 + top5[0].cpu().numpy()

#             val_loss = val_loss / (count + 1)
#             val_acc = val_acc / (count + 1)
#             val_acc5 = val_acc5 / (count + 1)

#             train_top1_history[epoch_counter] = train_acc
#             val_top1_history[epoch_counter] = val_acc
#             train_top5_history[epoch_counter] = train_acc5
#             val_top5_history[epoch_counter] = val_acc5
#             train_loss_history[epoch_counter] = train_loss
#             val_loss_history[epoch_counter] = val_loss

#             model_epochs[epoch_counter] = self.model
#             optimizer_epochs[epoch_counter] = self.optimizer

#             # warmup for the first 10 epochs
#             # if epoch_counter >= 10:
#             # No warmup
#             self.scheduler.step()
#             print(f"Epoch: {epoch_counter}   Train loss: {train_loss}   Top1 accuracy: {train_acc}   Top5 accuracy: {train_acc5}")
#             print(
#                 f"\tVal loss: {val_loss}   Top1 accuracy: {val_acc}   Top5 accuracy: {val_acc5}")
#             # print('learning rate', self.scheduler.get_lr()[0])
#             # print('logits:', logits[0])

#             if val_acc > best_acc:
#                 bad_count = 0
#                 best_loss = val_loss
#                 best_acc = val_acc
#                 best_epoch = epoch_counter
#             else:
#                 bad_count += 1

#             if bad_count > self.args.max_tol_pretrain:
#                 break

#             end_time = time.time()
#             print('time consumed:', end_time - start_time)

#         self.best_model = model_epochs[best_epoch]
#         self.best_optimizer = optimizer_epochs[best_epoch]

#         # save model checkpoints
#         checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(best_epoch)
#         torch.save({
#             'epoch': best_epoch,
#             'state_dict': self.best_model.state_dict(),
#             'optimizer': self.best_optimizer.state_dict(),
#         }, os.path.join(self.log_dir, checkpoint_name))

#         checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
#         torch.save({
#             'epoch': epoch_counter,
#             'state_dict': model_epochs[epoch_counter].state_dict(),
#             'optimizer': optimizer_epochs[epoch_counter].state_dict(),
#         }, os.path.join(self.log_dir, checkpoint_name))

#         print('best epoch: %d, train top1 acc:%.3f, top5 acc:%.3f; val top1 acc:%.3f, top5 acc:%.3f, train loss:%.4f, val loss: %.4f' % (
#             best_epoch, train_top1_history[best_epoch], train_top5_history[best_epoch],
#             val_top1_history[best_epoch], val_top5_history[best_epoch], 
#             train_loss_history[best_epoch], val_loss_history[best_epoch]))
#         return self.best_model, best_epoch, train_top1_history, val_top1_history, train_top5_history, val_top5_history, train_loss_history, val_loss_history


class SimCLRLoss(nn.Module):

    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.CEL = torch.nn.CrossEntropyLoss()
        self.device = torch.device('cpu')
    
    def to(self, device):
        self.device = device
        self.CEL = self.CEL.to(device)
        return self

    def info_nce_loss(self, features):
        
        device = self.device

        # print(features.shape)
        bs = int(features.shape[0] // 2)
        labels = torch.cat([torch.arange(bs) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        # # Normlize the features according to subject
        # if stratified == 'stratified':
        #     features_str = features.clone()
        #     features_str[:bs, :] = (features[:bs, :] -  features[:bs, :].mean(
        #         dim=0)) / (features[:bs, :].std(dim=0) + 1e-3)
        #     features_str[bs:, :] = (features[bs:, :] -  features[bs:, :].mean(
        #         dim=0)) / (features[bs:, :].std(dim=0) + 1e-3)
        #     features = F.normalize(features_str, dim=1)
        # elif stratified == 'bn':
        #     features_str = features.clone()
        #     features_str = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-3)
        #     features = F.normalize(features_str, dim=1)
        # elif stratified == 'no':
        #     features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # Put the positive column at the end (when all the entries are the same, the top1 acc will be 0; while if the
        # positive column is at the start, the top1 acc might be exaggerated)
        logits = torch.cat([negatives, positives], dim=1)
        # The label means the last column contain the positive pairs
        labels = torch.ones(logits.shape[0], dtype=torch.long)*(logits.shape[1]-1)
        labels = labels.to(device)

        logits = logits / self.temperature
        return logits, labels

    def forward(self, features):
        # fea need to be normalized to 1
        self.to(features.device)
        logits, labels = self.info_nce_loss(features)
        loss = self.CEL(logits, labels)
        return loss, logits, labels




class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = torch.device('cpu')

    def to(self, device):
        self.device = device
        return self

    def forward(self, features, labels=None, mask=None, modified=False):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].  need to be normalized to 1
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = self.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # features = F.normalize(features, dim=2)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob

        exp_logits = torch.exp(logits) * logits_mask

        if modified:
            nega_exp_logits_sum = (exp_logits*(~mask.bool())).sum(1)
            log_prob = torch.zeros_like(logits)
            for i in range(logits.shape[0]):
                for j in torch.nonzero(mask[i]).squeeze():
                    log_prob[i,j] = logits[i,j] - torch.log(nega_exp_logits_sum[i]+exp_logits[i,j])

        else:

            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss, exp_logits, mask