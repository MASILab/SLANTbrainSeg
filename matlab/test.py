import torch
import subjectlist as subl
import os
import argparse
import torchsrc

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)

parser = argparse.ArgumentParser()
# parser.add_argument('--task', required=True, help='STL_lmk | STL_clss | MTL_lmk | MTL_all ')
parser.add_argument('--piece', default='3_3_3', help='1_1_1 | 1_1_3 | 3_3_3 etc.')
# parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for, default=10')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--finetune',type=bool,default=False,help='fine tuning using true')
parser.add_argument('--fineepoch', type=int, default=5, help='fine tuning starting epoch')
parser.add_argument('--model_dir', help='where the model saved')
parser.add_argument('--test_img_dir', help='normalized image dir')
parser.add_argument('--out_dir', help='output dir')




opt = parser.parse_args()
print(opt)

# hyper parameters
epoch_num = 6
batch_size = 1
lmk_num = 133
learning_rate = opt.lr  #0.0001

finetune = opt.finetune
fineepoch = opt.fineepoch
model_dir = opt.model_dir
test_img_dir = opt.test_img_dir
out_dir = opt.out_dir

piece = opt.piece

piece_map = {}
piece_map['1_1_1'] = [0, 	96, 		0,	128, 		0,	88]
piece_map['3_1_1'] = [76,	172, 		0,	128, 		0,	88]
piece_map['1_3_1'] = [0,	96,			92,	220, 		0,	88]
piece_map['3_3_1'] = [76,	172, 		92,	220, 		0,	88]

piece_map['1_1_3'] = [0, 	96, 		0,	128, 		68,	156]
piece_map['3_1_3'] = [76,	172, 		0,	128, 		68,	156]
piece_map['1_3_3'] = [0,	96, 		92,	220, 		68,	156]
piece_map['3_3_3'] = [76,	172, 		92,	220, 		68,	156]

#middle ones
piece_map['2_1_1'] = [38, 	134, 		0,	128, 		0,	88]
piece_map['2_3_1'] = [38, 	134,		92,	220, 		0,	88]
piece_map['2_1_3'] = [38, 	134, 		0,	128, 		68,	156]
piece_map['2_3_3'] = [38, 	134, 		92,	220, 		68,	156]

piece_map['1_2_1'] = [0, 	96, 		46,	174, 		0,	88]
piece_map['3_2_1'] = [76,	172, 		46,	174, 		0,	88]
piece_map['1_2_3'] = [0, 	96, 		46,	174, 		68,	156]
piece_map['3_2_3'] = [76,	172, 		46,	174, 		68,	156]

piece_map['1_1_2'] = [0, 	96, 		0,	128, 		34,	122]
piece_map['3_1_2'] = [76,	172, 		0,	128, 		34,	122]
piece_map['1_3_2'] = [0,	96,			92,	220, 		34,	122]
piece_map['3_3_2'] = [76,	172, 		92,	220, 		34,	122]

piece_map['1_2_2'] = [0, 	96, 		46,	174, 		34,	122]
piece_map['3_2_2'] = [76,	172, 		46,	174, 		34,	122]
piece_map['2_2_2'] = [38, 	134, 		46,	174, 		34,	122]
piece_map['2_1_2'] = [38, 	134, 		0,	128, 		34,	122]
piece_map['2_3_2'] = [38, 	134, 		92,	220, 		34,	122]
piece_map['2_2_1'] = [38, 	134, 		46,	174, 		0,	88]
piece_map['2_2_3'] = [38, 	134, 		46,	174, 		68,	156]



# define paths
# train_img_dir = '/fs4/masi/huoy1/FS3_backup/software/full-multi-atlas/atlas-processing/aladin-reg-images-normalized'
# train_seg_dir = '/fs4/masi/huoy1/FS3_backup/software/full-multi-atlas/atlas-processing/aladin-reg-labels'
# test_img_dir = '/fs4/masi/huoy1/FS3_backup/software/full-multi-atlas/atlas-processing/aladin-reg-images-normalized'
# test_seg_dir = '/fs4/masi/huoy1/FS3_backup/software/full-multi-atlas/atlas-processing/aladin-reg-labels'


# train_img_dir = '/share3/huoy1/3DUnet/brainCOLOR/reample_train/aladin-reg-images-normalized'
# train_seg_dir = '/share3/huoy1/3DUnet/brainCOLOR/reample_train/aladin-reg-labels'

# # 5000 MAS
# test_img_dir = '/share3/huoy1/3DUnet/brainCOLOR/reample_train/aladin-reg-images-normalized'
# test_seg_dir = '/share3/huoy1/3DUnet/brainCOLOR/reample_train/aladin-reg-labels'
# train_list_file = '/share4/huoy1/Deep_5000_Brain/sublist/sublist_5k.txt'
# working_dir = '/share4/huoy1/Deep_5000_Brain/working_dir/'


# 1_1_1
# test_img_dir = '/share3/huoy1/3DUnet/brainCOLOR/reample_train/aladin-reg-images-normalized'
# test_seg_dir = '/share3/huoy1/3DUnet/brainCOLOR/reample_train/aladin-reg-labels'
# train_list_file = '/share4/huoy1/Deep_5000_Brain/sublist/sublist_mni.txt'

# model_dir = os.path.join('/share4/huoy1/Deep_5000_Brain/working_dir/',piece)
train_root_dir = os.path.join(model_dir, piece,'test_out')
# test_img_dir = '/share4/huoy1/BaxterDeepBrainCOLOR/working_dir/deep_learning'



# make img list


# out = os.path.join('/share4/huoy1/BaxterDeepBrainCOLOR/working_dir/all_piece',piece,'test_out')
out = os.path.join(out_dir,piece,'test_out')
mkdir(out)

test_img_subs,test_img_files = subl.get_sub_list(test_img_dir)
test_dict = {}
test_dict['img_subs'] = test_img_subs
test_dict['img_files'] = test_img_files



# load image
# train_set = torchsrc.imgloaders.pytorch_loader_allpiece(train_dict,num_labels=lmk_num,piece=piece,piece_map=piece_map)
# train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=1)
test_set = torchsrc.imgloaders.pytorch_loader_allpiece(test_dict,num_labels=lmk_num,piece=piece,piece_map=piece_map)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=1)

# load network
model = torchsrc.models.UNet3D(in_channel=1, n_classes=lmk_num)
# model = torchsrc.models.VNet()

# print_network(model)
#
# load optimizor
optim = torch.optim.Adam(model.parameters(), lr = learning_rate, betas=(0.9, 0.999))
# optim = torch.optim.SGD(model.parameters(), lr=learning_curve() _rate, momentum=0.9)

# load CUDA
cuda = torch.cuda.is_available()
torch.manual_seed(1)
if cuda:
	torch.cuda.manual_seed(1)
	model = model.cuda()
print("finsih cuda")

# load trainer
trainer = torchsrc.Trainer(
	cuda=cuda,
	model=model,
	optimizer=optim,
	train_loader=[],
	test_loader=test_loader,
	train_root_dir = train_root_dir,
	out=out,
	max_epoch = epoch_num,
	batch_size = batch_size,
	lmk_num = lmk_num,
	finetune = finetune,
	fineepoch = fineepoch
)


print("==start training==")

start_epoch = 5
start_iteration = 1
trainer.epoch = start_epoch
trainer.iteration = start_iteration
trainer.test_epoch()







