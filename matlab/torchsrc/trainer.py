import datetime
import math
import os
import os.path as osp
import shutil

#import fcn
import numpy as np
import pytz
import scipy.misc
import scipy.io as sio
import nibabel as nib
from scipy.spatial import distance
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm
import skimage
import random
from utils.image_pool import ImagePool
import torchsrc

def saveOneImg(img,path,cate_name,sub_name,surfix,):
    filename = "%s-x-%s-x-%s.png"%(cate_name,sub_name,surfix)
    file = os.path.join(path,filename)
    scipy.misc.imsave(file, img)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols

def ind2sub(array_shape, ind):
    rows = (ind.astype('int') / array_shape[1])
    cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return (rows, cols)


def weighted_center(input,threshold=0.75):
    # m= torch.nn.Tanh()
    # input = m(input)

    input = torch.add(input, -input.min().expand(input.size())) / torch.add(input.max().expand(input.size()), -input.min().expand(input.size()))
    m = torch.nn.Threshold(threshold, 0)
    input = m(input)
    # if input.sum()==0:
    #     input=input
    # mask_ind = input.le(0.5)
    # input.masked_fill_(mask_ind, 0.0)
    grid = np.meshgrid(range(input.size()[0]), range(input.size()[1]), indexing='ij')
    x0 = torch.mul(input, Variable(torch.from_numpy(grid[1]).float().cuda())).sum() / input.sum()
    y0 = torch.mul(input, Variable(torch.from_numpy(grid[0]).float().cuda())).sum() / input.sum()
    return x0, y0


# def max_center(input,target,pts):
#     input.max()
#     return x0, y0


def get_distance(target,score,ind,Threshold=0.75):
    dist_list = []
    coord_list = []
    target_coord_list = []
    weight_coord_list = []
    for i in range(target.size()[1]):
        targetImg = target[ind,i,:,:].data.cpu().numpy()
        scoreImg = score[ind,i,:,:].data.cpu().numpy()
        targetCoord = np.unravel_index(targetImg.argmax(),targetImg.shape)
        scoreCoord = np.unravel_index(scoreImg.argmax(),scoreImg.shape)
        # grid = np.meshgrid(range(score.size()[2]), range(score.size()[3]), indexing='ij')
        # x0 = torch.mul(score[ind, i, :, :], Variable(torch.from_numpy(grid[0]).float().cuda())).sum() / score[ind, i, :,
        #                                                                                               :].sum()
        # y0 = torch.mul(score[ind, i, :, :], Variable(torch.from_numpy(grid[1]).float().cuda())).sum() / score[ind, i, :,
        #                                                                                               :].sum()
        #
        y0,x0 = weighted_center(score[ind,i,:,:],Threshold)

        weightCoord = (x0.data.cpu().numpy()[0],y0.data.cpu().numpy()[0])
        distVal = distance.euclidean(scoreCoord,targetCoord)
        dist_list.append(distVal)
        coord_list.append(scoreCoord)
        target_coord_list.append(targetCoord)
        weight_coord_list.append(weightCoord)
    return dist_list,coord_list,target_coord_list,weight_coord_list

def dice_loss(input, target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    # uniques = np.unique(target.numpy())
    # assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

    probs = F.softmax(input)
    num = probs * target  # b,c,h,w--p*g
    num = torch.sum(num, dim=2)
    num = torch.sum(num, dim=3)  # b,c

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=2)
    den1 = torch.sum(den1, dim=3)  # b,c,1,1

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=2)
    den2 = torch.sum(den2, dim=3)  # b,c,1,1

    dice = 2 * ((num+0.0000001) / (den1 + den2+0.0000001))
    dice_eso = dice[:, 1]  # we ignore bg dice val, and take the fg

    dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz

    return dice_total

def dice_loss_norm(input, target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    # uniques = np.unique(target.numpy())
    # assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

    probs = F.softmax(input)
    num = probs * target  # b,c,h,w--p*g
    num = torch.sum(num, dim=3)
    num = torch.sum(num, dim=2)  #
    num = torch.sum(num, dim=0)# b,c

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)
    den1 = torch.sum(den1, dim=2)  # b,c,1,1
    den1 = torch.sum(den1, dim=0)

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=3)
    den2 = torch.sum(den2, dim=2)  # b,c,1,1
    den2 = torch.sum(den2, dim=0)

    dice = 2 * ((num+0.0000001) / (den1 + den2+0.0000001))
    dice_eso = dice[1:]  # we ignore bg dice val, and take the fg
    dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz
    return dice_total




def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


def l2_normloss(input,target,size_average=True):
    criterion = torch.nn.MSELoss().cuda()  
    loss = criterion(input, target)
    # if size_average:
    #     loss /= (target.size()[0]*target.size()[1])
    return loss

def l2_normloss_new(input,target,mask):
    loss = input - target
    loss = torch.pow(loss,2)
    loss = torch.mul(loss, mask)
    loss = loss.sum() / mask.sum()
    return loss

def l1_normloss(input,target,size_average=True):
    criterion = torch.nn.L1Loss().cuda()
    loss = criterion(input, target)
    # if size_average:
    #     loss /= (target.size()[0]*target.size()[1])
    return loss


def l1_smooth_normloss(input,target,size_average=True):
    criterion = torch.nn.SmoothL1Loss().cuda()
    loss = criterion(input, target)
    # if size_average:
    #     loss /= (target.size()[0]*target.size()[1])
    return loss


def l2_normloss_compete(input,target,size_average=True):
    mask = torch.sum(target, 1)
    mask = mask.expand(input.size())
    mask_ind = mask.le(0.5)
    input.masked_fill_(mask_ind, 0.0)
    mask = torch.mul(mask, 0)
    input = torch.mul(input,10)
    criterion = torch.nn.MSELoss().cuda()
    loss = criterion(input,mask)
    return loss

def l2_normloss_all(inputs,target,category_name,all_categories):
    for i in range(len(all_categories)):
        cate = all_categories[i]
        if i == 0 :
            if category_name == cate:
                loss = l2_normloss(inputs[i],target)
            else :
                loss = l2_normloss_compete(inputs[i],target)
        else:
            if category_name == cate :
                loss += l2_normloss(inputs[i],target)
            else :
                loss += l2_normloss_compete(inputs[i],target)
    return loss



def mse_loss(input, target):
    return torch.sum((input - target) ** 2)


def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)


def write_log(log_file,target,pred_lmk,pts,epoch,batch_idx,sub_name,category_name,Threshold = 0.75):
    if not (Threshold == 0.75):
        log_file = log_file.replace('log.txt', ('log_%.2f' % Threshold))

    if batch_idx == 0 and os.path.exists(log_file):
        os.remove(log_file)

    fv = open(log_file, 'a')
    for bi in range(target.size()[0]):
        distance_list, coord_list, target_coord_list, weight_coord_list = get_distance(target, pred_lmk, bi,Threshold)
        show_str = ''
        for di in range(pts[bi].size()[0]):
            if (sum(sum(pts[0] == -1)) == 0):
                show_str = show_str + ', dist[%d]=%.4f,predlmk[%d]=(%.4f;%.4f),truelmk[%d]=(%.4f;%.4f),weightlmk[%d]=(%.4f;%.4f)' % (di,
                        distance_list[di], di,coord_list[di][1],coord_list[di][0],di, pts[bi][di, 0], pts[bi][di, 1],di,weight_coord_list[di][1],weight_coord_list[di][0])
        fv.write('epoch=%d,batch_idx=%d, subject=%s, category=%s, %s\n' % (
        epoch, batch_idx, sub_name, category_name, show_str))
    fv.close()


def save_images(results_epoch_dir,data,sub_name,cate_name,pred_lmk,target=None):
    saveOneImg(data[0, 0, :, :].data.cpu().numpy(), results_epoch_dir, cate_name,sub_name, "_trueGray")
    for i in range(pred_lmk.size()[1]):
        saveOneImg(pred_lmk[0, i, :, :].data.cpu().numpy(), results_epoch_dir, cate_name,sub_name, "_pred%d" % (i))
        if not (target is None):
            saveOneImg(target[0, i, :, :].data.cpu().numpy(), results_epoch_dir, cate_name,sub_name, "_true%d" % (i))




def prior_loss(input,category_name,pts,target):
    mu = {}
    std = {}
    #caculated from get_spatial_prior
    # mu['KidneyLong'] = [210.420535]
    # std['KidneyLong'] = [25.846215]
    # mu['KidneyTrans'] = [104.701820, 96.639190]
    # std['KidneyTrans'] = [17.741928, 19.972482]
    # mu['LiverLong'] = [303.206934]
    # std['LiverLong'] = [45.080338]
    # mu['SpleenLong'] = [202.573985]
    # std['SpleenLong'] = [39.253982]
    # mu['SpleenTrans'] = [190.321392, 86.738878]
    # std['SpleenTrans'] = [41.459823, 21.711744]

    pts = Variable(pts.cuda())
    # for i in input

    # grid = np.meshgrid(range(input.size()[2]), range(input.size()[3]), indexing='ij')
    x0, y0 = weighted_center(input[0, 0, :, :])
    x1, y1 = weighted_center(input[0, 1, :, :])

    dist = torch.sqrt(torch.pow(x0-x1, 2)+torch.pow(y0-y1, 2))
    truedist = torch.sqrt(torch.pow(pts[0,0,0]-pts[0,1,0], 2)+torch.pow(pts[0,0,1]-pts[0,1,1], 2))
    loss = torch.abs(dist-truedist)
    #
    if category_name == 'KidneyTrans' or category_name == 'SpleenTrans':
    #     # x2 = torch.mul(input[0, 2, :, :], Variable(torch.from_numpy(grid[1]).float().cuda())).sum()/input[0, 2, :, :].sum()
    #     # y2 = torch.mul(input[0, 2, :, :], Variable(torch.from_numpy(grid[0]).float().cuda())).sum()/input[0, 2, :, :].sum()
    #     # x3 = torch.mul(input[0, 3, :, :], Variable(torch.from_numpy(grid[1]).float().cuda())).sum()/input[0, 3, :, :].sum()
    #     # y3 = torch.mul(input[0, 3, :, :], Variable(torch.from_numpy(grid[0]).float().cuda())).sum()/input[0, 3, :, :].sum()

        # dist2 = torch.sqrt(torch.pow(x2 - x3, 2) + torch.pow(y2 - y3, 2))
        # loss += torch.abs(dist2-mu[category_name][1])
        x2, y2 = weighted_center(input[0, 2, :, :])
        x3, y3 = weighted_center(input[0, 3, :, :])
        dist = torch.sqrt(torch.pow(x2-x3, 2)+torch.pow(y2-y3, 2))
        truedist = torch.sqrt(torch.pow(pts[0,2,0]-pts[0,3,0], 2)+torch.pow(pts[0,2,1]-pts[0,3,1], 2))
        loss += torch.abs(dist-truedist)
    # # criterion = torch.nn.L1Loss().cuda()
    # # loss = criterion(dist,mu[category_name][0])

    return loss

def dice_error(input, target):
    eps = 0.000001
    _, result_ = input.max(1)

    _, target_ = target.max(1)

    result_ = torch.squeeze(result_)
    target_ = torch.squeeze(target_)
    if input.is_cuda:
        result = torch.cuda.FloatTensor(result_.size())
        target = torch.cuda.FloatTensor(target_.size())
    else:
        result = torch.FloatTensor(result_.size())
        target = torch.FloatTensor(target_.size())
    result.copy_(result_.data)
    target.copy_(target_.data)
    result = result.view(-1)
    target = target.view(-1)
    intersect = torch.dot(result, target)

    result_sum = torch.sum(result)
    target_sum = torch.sum(target)
    union = result_sum + target_sum + 2*eps
    intersect = np.max([eps, intersect])
    # the target volume can be empty - so we still want to
    # end up with a score of 1 if the result is 0/0
    IoU = intersect / union
#    print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
#        union, intersect, target_sum, result_sum, 2*IoU))
    return 2*IoU

def dice_loss_3d(input, target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 5, "Input must be a 5D Tensor."
    # uniques = np.unique(target.numpy())
    # assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"
    target = target.view(target.size(0), target.size(1), target.size(2), -1)
    input = input.view(input.size(0), input.size(1), input.size(2), -1)
    probs = F.softmax(input)

    num = probs * target  # b,c,h,w--p*g
    num = torch.sum(num, dim=3)
    num = torch.sum(num, dim=2)  #
    num = torch.sum(num, dim=0)# b,c

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)
    den1 = torch.sum(den1, dim=2)  # b,c,1,1
    den1 = torch.sum(den1, dim=0)

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=3)
    den2 = torch.sum(den2, dim=2)  # b,c,1,1
    den2 = torch.sum(den2, dim=0)

    dice = 2 * (num / (den1 + den2+0.0000001))
    dice_eso = dice[0:]  # we ignore bg dice val, and take the fg
    dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz
    dice_total = dice_total
    return dice_total





def dice_l2(input,target,size_average=True):
    criterion = torch.nn.MSELoss().cuda()
    loss = criterion(input, target)
    # if size_average:
    #     loss /= (target.size()[0]*target.size()[1])
    return loss

class Trainer(object):

    def __init__(self, cuda, model, optimizer=None,
                train_loader=None,test_loader=None,lmk_num=None,
                train_root_dir=None,out=None, max_epoch=None, batch_size=None,
                size_average=False, interval_validate=None,	fineepoch=None,
	            finetune=False, compete = False,onlyEval=False):
        self.cuda = cuda

        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.interval_validate = interval_validate

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        self.size_average = size_average

        self.train_root_dir = train_root_dir
        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.lmk_num = lmk_num


        self.max_epoch = max_epoch
        self.epoch = 0
        self.iteration = 0
        self.best_mean_iu = 0
        self.batch_size = batch_size

        self.finetune = finetune
        self.fineepoch = fineepoch


    def forward_step(self, data, category_name):
        if category_name == 'KidneyLong':
            pred_lmk = self.model(data, 'KidneyLong')
        elif category_name == 'KidneyTrans':
            pred_lmk = self.model(data, 'KidneyTrans')
        elif category_name == 'LiverLong':
            pred_lmk = self.model(data, 'LiverLong')
        elif category_name == 'SpleenLong':
            pred_lmk = self.model(data, 'SpleenLong')
        elif category_name == 'SpleenTrans':
            pred_lmk = self.model(data, 'SpleenTrans')
        return pred_lmk

    def validate(self):
        self.model.train()
        out = osp.join(self.out, 'seg_output')
        out_vis = osp.join(self.out, 'visualization')
        results_epoch_dir = osp.join(out,'epoch_%04d' % self.epoch)
        mkdir(results_epoch_dir)

        for batch_idx, (data,target,sub_name) in tqdm.tqdm(
                # enumerate(self.test_loader), total=len(self.test_loader),
                enumerate(self.test_loader), total=len(self.test_loader),
                desc='Valid epoch=%d' % self.epoch, ncols=80,
                leave=False):

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data,volatile=True), Variable(target,volatile=True)

            pred = self.model(data)

            lbl_pred = pred.data.max(1)[1].cpu().numpy()[:,:, :].astype('uint8')
            batch_num = lbl_pred.shape[0]
            for si in range(batch_num):
                curr_sub_name = sub_name[si]
                out_img_dir = os.path.join(results_epoch_dir, 'seg')
                mkdir(out_img_dir)
                out_nii_file = os.path.join(out_img_dir,('%s_seg.nii.gz'%(curr_sub_name)))
                seg_img = nib.Nifti1Image(lbl_pred[si], affine=np.eye(4))
                nib.save(seg_img, out_nii_file)

            # if self.epoch==0:
            #     lbl_target = target.data.max(1)[1].cpu().numpy()[:,:, :].astype('uint8')
            #     batch_num = lbl_target.shape[0]
            #     for si in range(batch_num):
            #         curr_sub_name = sub_name[si]
            #         out_img_dir = os.path.join(results_epoch_dir, 'true')
            #         mkdir(out_img_dir)
            #         out_nii_file = os.path.join(out_img_dir,('%s_true.nii.gz'%(curr_sub_name)))
            #         seg_img = nib.Nifti1Image(lbl_target[si], affine=np.eye(4))
            #         nib.save(seg_img, out_nii_file)


            #     lbl_img = data.data.cpu().numpy()
            #     batch_num = lbl_img.shape[0]
            #     for si in range(batch_num):
            #         curr_sub_name = sub_name[si]
            #         out_img_dir = os.path.join(results_epoch_dir, 'img')
            #         mkdir(out_img_dir)
            #         out_nii_file = os.path.join(out_img_dir,('%s_img.nii.gz'%(curr_sub_name)))
            #         seg_img = nib.Nifti1Image(lbl_img[si,0], affine=np.eye(4))
            #         nib.save(seg_img, out_nii_file)


    def train(self):
        self.model.train()
        out = osp.join(self.out, 'visualization')
        mkdir(out)
        log_file = osp.join(out, 'training_loss.txt')
        fv = open(log_file, 'a')

        for batch_idx, (data, target, sub_name) in tqdm.tqdm(
            enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
  
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            pred = self.model(data)
            self.optim.zero_grad()


            # loss = dice_loss_3d(pred*100 ,target)
            # print('epoch=%d, batch_idx=%d, loss=%.4f \n'%(self.epoch,batch_idx,loss.data[0]))
            # fv.write('epoch=%d, batch_idx=%d, loss=%.4f \n'%(self.epoch,batch_idx,loss.data[0]))

            loss = dice_loss_3d(pred*100 ,target)
            # loss_l2 = dice_l2(pred*10000 ,target*10000)/1000
            # print('epoch=%d, batch_idx=%d, loss=%.4f, loss l2=%.4f \n'%(self.epoch,batch_idx,loss.data[0],loss_l2.data[0]))
            # fv.write('epoch=%d, batch_idx=%d, loss=%.4f, loss l2=%.4f \n'%(self.epoch,batch_idx,loss.data[0],loss_l2.data[0]))
            # loss = dice_error(pred,target)
            print('epoch=%d, batch_idx=%d, loss=%.4f \n'%(self.epoch,batch_idx,loss.data[0]))
            fv.write('epoch=%d, batch_idx=%d, loss=%.4f \n'%(self.epoch,batch_idx,loss.data[0]))
            # loss = loss+loss_l2
            loss.backward()
            self.optim.step()

        fv.close()

    def train_epoch(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            out = osp.join(self.out, 'models')
            mkdir(out)

            model_pth = '%s/model_epoch_%04d.pth' % (out, epoch)

            if self.finetune:
                old_out = out.replace('finetune_out','test_out')
                old_model_pth = '%s/model_epoch_%04d.pth' % (old_out, self.fineepoch)
                self.model.load_state_dict(torch.load(old_model_pth))
	

            if os.path.exists(model_pth):
                print("start load")
                self.model.load_state_dict(torch.load(model_pth))\
                #print("finsih load") 
                # self.validate()
            else:
                self.train()
                # if epoch % 20 == 0:
                self.validate()
                torch.save(self.model.state_dict(), model_pth)

                # torch.save(self.model.state_dict(), model_pth)

    def test_epoch(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Test', ncols=80):
            self.epoch = epoch
            train_root_dir = osp.join(self.train_root_dir, 'models')

            model_pth = '%s/model_epoch_%04d.pth' % (train_root_dir, epoch)
            if os.path.exists(model_pth):
                self.model.load_state_dict(torch.load(model_pth))
                self.validate()

             



