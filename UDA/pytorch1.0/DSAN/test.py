from __future__ import print_function
import torch
import logging
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader
from torch import optim
import ResNetb as models
from Weight import Weight
from Config import *
import time
import data_loader
import mmdb

soft_label = False

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
logging.getLogger().setLevel(logging.INFO)
cuda = not no_cuda and torch.cuda.is_available()
#torch.manual_seed(seed)
#if cuda:
#    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

data_s = data_loader.DealDataset1()
dataloader_source = data_loader.DataLoader(data_s, batch_size, shuffle=True, num_workers=0)
data_s_test = data_loader.DealDataset2()
dataloader_source_test = data_loader.DataLoader(data_s_test, batch_size, shuffle=True, num_workers=0)
data_t = data_loader.DealDataset()
dataloader_target = data_loader.DataLoader(data_t, batch_size, shuffle=True, num_workers=0)
data_t_test = data_loader.DealDataset3()
dataloader_target_test = data_loader.DataLoader(data_t_test, batch_size, shuffle=True, num_workers=0)
data_t_0 = data_loader.DealDataset4()
dataloader_target_0 = data_loader.DataLoader(data_t_0, batch_size, shuffle=True, num_workers=0)

len_source_dataset = len(dataloader_source.dataset)
len_target_dataset = len(dataloader_target.dataset)
len_source_loader = len(dataloader_source)
len_target_loader = len(dataloader_target)
len_target_0_loder = len(dataloader_target_0)



def train(epoch, model):
    # LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
    # print('learning rate{: .4f}'.format(LEARNING_RATE) )
    # if bottle_neck:
    #     optimizer = torch.optim.SGD([
    #         {'params': model.feature_layers.parameters(), 'lr': LEARNING_RATE},
    #         {'params': model.bottle.parameters(), 'lr': LEARNING_RATE},
    #         {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
    #     ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)
    # else:
    #     optimizer = torch.optim.SGD([
    #         {'params': model.feature_layers.parameters()},
    #         {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
    #         ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

    model.train()

    # for i, ((data_source, label_source), (data_target, label_target), (data_target_0, label_target_0)) in enumerate(
    #         zip(dataloader_source, dataloader_target, dataloader_target_0)):

    iter_source = iter(dataloader_source)
    iter_target = iter(dataloader_target)
    iter_target_0 = iter(dataloader_target_0)

    num_iter = len_source_loader
    epoch_acc = 0

    # for batch_idx, (x_data, y_data) in enumerate(dataloader_source):
    #     data_source = torch.Tensor(x_data).cuda()
    #     label_source = y_data.cuda()

    for i in range(1, num_iter):
        data_source, label_source= iter_source.next()
        data_target, label_target = iter_target.next()


        if i % len_target_0_loder == 0:
            iter_target_0 = iter(dataloader_target_0)
            data_target_0, label_target_0 = iter_target_0.next()

        else:
            data_target_0, label_target_0 = iter_target_0.next()

        if cuda:
            data_source, label_source = data_source.cuda(), label_source.cuda()
            data_target, label_target = data_target.cuda(), label_target.cuda()
            data_target_0, label_target_0 = data_target_0.cuda(), label_target_0.cuda()

        data_source, label_source = Variable(data_source), Variable(label_source)
        data_target, label_target = Variable(data_target), Variable(label_target)
        data_target_0, label_target_0 = Variable(data_target_0), Variable(label_target_0)

        label_source_pred, loss_mmd = model(data_source, data_target, label_source)
        loss_cls = F.nll_loss(F.log_softmax(label_source_pred, dim=1), label_source)

        lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1

        if soft_label:
            #soft loss
            #hyperparam
            nu = 0.01
            temperature = 2
            if cuda:
                encoder = model.feature_layers.cuda()
                bottle = model.bottle.cuda()
                classifier = model.cls_fc.cuda()

            soft_labels = mmdb.gen_soft_labels(9, dataloader_source, model.feature_layers, model.bottle, model.cls_fc)

            soft_label_for_batch = mmdb.ret_soft_label(label_target_0, soft_labels)

            soft_label_for_batch = soft_label_for_batch.cuda()
            soft_label_for_batch = Variable(soft_label_for_batch)

            tgt_output_cl_feature = encoder(data_target_0)
            tgt_output_cl_features = bottle(tgt_output_cl_feature)
            tgt_output_cl_0 = classifier(tgt_output_cl_features)

            loss_cls_0 = F.nll_loss(F.log_softmax(tgt_output_cl_0, dim=1), label_target_0)


            output_cl_score_0 = F.softmax(tgt_output_cl_0 / temperature, dim=1)

            loss_soft = - (torch.sum(soft_label_for_batch * torch.log(output_cl_score_0))) / float(
                output_cl_score_0.size(0))

            loss = loss_cls + param * lambd * loss_mmd +  loss_cls_0

        else:
             loss = loss_cls + param * lambd * loss_mmd

        # criterion = torch.nn.CrossEntropyLoss()
        # logit = model(data_source)
        # loss = criterion(logit, label_source)
        pred = label_source_pred.argmax(dim=1)
        correct = torch.eq(pred,label_source).float().sum().item()
        epoch_acc += correct

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if i % log_interval == 0:
    epoch_acc = epoch_acc / len(dataloader_source.dataset)
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}'.format(
        epoch, i * len(data_source), len_source_dataset,
        100. * i / len_source_loader, loss.item(), loss_cls.item(), loss_mmd.item()))
    # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss: {:.6f}\tcorrect: {:.6f}'.format(
    #     epoch, i * len(data_source), len_source_dataset,
    #     100. * i / len_source_loader, loss, loss,  epoch_acc))


def test(model):
    model.eval()
    test_loss = 0
    epoch_acc = 0
    correct_all = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader_target_test:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            s_output, t_output = model(data, data, target)
            test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target).item() # sum up batch loss
            pred = s_output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            # logit = model(data)
            # pred = logit.argmax(dim=1)
            # correct = torch.eq(pred, target).float().sum().item()
            # correct_all += correct
            # epoch_acc += correct
        test_loss /= len_target_dataset
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            target_name, test_loss, correct, len(dataloader_target_test.dataset),
            100. * correct / len(dataloader_target_test.dataset)))
        # epoch_acc = epoch_acc / len(dataloader_target_test.dataset)
        # print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f})\n'.format(
        #     target_name, test_loss, correct_all, len(dataloader_target_test.dataset),
        #     epoch_acc))
    return correct


if __name__ == '__main__':
    model = models.DSAN(num_classes=class_num)
    #model = models.CNNbase()
    correct = 0
    steps =[50,150]
    step = 0
    print(model)
    if cuda:
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=0.1)
    time_start=time.time()
    for epoch in range(1, epochs + 1):
        logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, epochs - 1) + '-' * 5)
        # Update the learning rate
        if lr_scheduler is not None:
            # self.lr_scheduler.step(epoch)
            logging.info('current lr: {}'.format(lr_scheduler.get_last_lr()))
        else:
            logging.info('current lr: {}'.format(lr))
        train(epoch, model)
        t_correct = test(model)
        if t_correct > correct:
            correct = t_correct
            torch.save(model, 'model.pkl')
        end_time = time.time()
        step = step + 1
        lr_scheduler.step()
        # print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
        #       source_name, target_name, correct, 100. * correct / len_target_dataset ))
        # print('cost time:', end_time - time_start)
