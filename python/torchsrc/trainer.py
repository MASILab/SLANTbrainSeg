import datetime
import os
import os.path as osp

import numpy as np
import pytz
import scipy.misc
import nibabel as nib
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm


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




def save_images(results_epoch_dir,data,sub_name,cate_name,pred_lmk,target=None):
    saveOneImg(data[0, 0, :, :].data.cpu().numpy(), results_epoch_dir, cate_name,sub_name, "_trueGray")
    for i in range(pred_lmk.size()[1]):
        saveOneImg(pred_lmk[0, i, :, :].data.cpu().numpy(), results_epoch_dir, cate_name,sub_name, "_pred%d" % (i))
        if not (target is None):
            saveOneImg(target[0, i, :, :].data.cpu().numpy(), results_epoch_dir, cate_name,sub_name, "_true%d" % (i))







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

             



