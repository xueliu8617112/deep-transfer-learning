#!/usr/bin/env python
# encoding: utf-8
import torch
from Weight import Weight
import torch.nn.functional as F
from torch.autograd import Variable


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    :param source: 源域数据，行表示样本数目，列表示样本数据维度
    :param target: 目标域数据 同source
    :param kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
    :param kernel_num:  取不同高斯核的数量
    :param fix_sigma:是否固定，如果固定，则为单核MMD
    :return:  (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
						矩阵，表达形式:
						[	K_ss K_st
							K_ts K_tt ]
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0) # 合并在一起
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) # 计算高斯核中的|x-y|
    # 计算多核中每个核的bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    # 高斯核的公式，exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)  # 将多个核合并在一起

def lmmd(source, target, s_label, t_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = source.size()[0]
    weight_ss, weight_tt, weight_st = Weight.cal_weight(s_label, t_label, type='visual', class_num=9)
    weight_ss = torch.from_numpy(weight_ss).cuda()
    weight_tt = torch.from_numpy(weight_tt).cuda()
    weight_st = torch.from_numpy(weight_st).cuda()

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = torch.Tensor([0]).cuda()
    if torch.sum(torch.isnan(sum(kernels))):
        return loss
    SS = kernels[:batch_size, :batch_size]
    TT = kernels[batch_size:, batch_size:]
    ST = kernels[:batch_size, batch_size:]

    loss += torch.sum( weight_ss * SS + weight_tt * TT - 2 * weight_st * ST )
    return loss

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul,
                             	kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size] # Source<->Source
    YY = kernels[batch_size:, batch_size:] # Target<->Target
    XY = kernels[:batch_size, batch_size:] # Source<->Target
    YX = kernels[batch_size:, :batch_size] # Target<->Source
    loss = torch.mean(XX + YY - XY -YX) # 这里是假定X和Y的样本数量是相同的 # 当不同的时候，就需要乘上上面的M矩阵
    return loss


def gen_soft_labels(num_classes, src_train_loader, src_encoder, src_bottle, src_classifier):
    cuda = torch.cuda.is_available()
    temperature = 2

    soft_labels = torch.zeros(num_classes, 1, num_classes)
    sum_classes = torch.zeros(num_classes)
    pred_scores_total = []
    label_total = []
    if cuda:
        src_encoder = src_encoder.cuda()
        src_bottle =  src_bottle.cuda()
        src_classifier = src_classifier.cuda()

    for batch_idx, (src_data, label) in enumerate(src_train_loader):
        label_total.append(label)
        if cuda:
            src_data, label = src_data.cuda(), label.cuda()
            src_data, label = Variable(src_data), Variable(label)

        src_feature_ = src_encoder(src_data)
        src_feature = src_bottle(src_feature_)
        output = src_classifier(src_feature)

        pred_scores = F.softmax(output / temperature, dim=1).data.cpu()
        pred_scores_total.append(pred_scores)

    pred_scores_total = torch.cat(pred_scores_total)
    label_total = torch.cat(label_total)

    # sum of each class
    for i in range(len(src_train_loader.dataset)):
        sum_classes[label_total[i]] += 1  # 计算每一个位置的label数量
        soft_labels[label_total[i]][0] += pred_scores_total[i]  # 第label_total[i] 填入预测值， 10 [1 10]>predicted output
    # average
    for cl_idx in range(num_classes):
        soft_labels[cl_idx][0] /= sum_classes[cl_idx]
    return soft_labels


def ret_soft_label(label, soft_labels):
    num_classes = 9
    soft_label_for_batch = torch.zeros(label.size(0), num_classes)
    for i in range(label.size(0)):
        soft_label_for_batch[i] = soft_labels[label.data[i]]

    return soft_label_for_batch

if __name__ == "__main__":
    import numpy as np
    data_1 = torch.tensor(np.random.normal(0,10,(100,50)))
    data_2 = torch.tensor(np.random.normal(10,10,(100,50)))

    print("MMD Loss:",mmd(data_1,data_2))

    data_1 = torch.tensor(np.random.normal(0,10,(100,50)))
    data_2 = torch.tensor(np.random.normal(0,9,(100,50)))

    print("MMD Loss:",mmd(data_1,data_2))