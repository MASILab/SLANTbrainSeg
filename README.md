# SLANT: Deep Whole Brain High Resolution Segmentation 
### [[PyTorch]](https://github.com/MASILab/SLANTbrainSeg/tree/master/python) [[project page]](https://github.com/MASILab/SLANTbrainSeg/)   [[paper]](https://arxiv.org/pdf/1806.00546.pdf)

A T1 MRI scan can be segmented to 133 labels based on BrainCOLOR protocal(http://braincolor.mindboggle.info/protocols/).
<img src="https://github.com/MASILab/SLANTbrainSeg/blob/master/screenshot/test_volume_result.jpg" width="600px"/>

It has been implemented as a single Docker.
```diff
- Please cite the following MICCAI/NeuroImage paper, if you used the SLANT whole brain segmentation.
```
The papers can be found [SLANT](https://arxiv.org/pdf/1806.00546.pdf),[NeuroImage](https://www.sciencedirect.com/science/article/pii/S1053811919302307), whole full citation are

Yuankai Huo, Zhoubing Xu, Yunxi Xiong, Katherine Aboud, Parasanna Parvathaneni, Shunxing Bao, Camilo Bermudez, Susan M. Resnick, Laurie E. Cutting, and Bennett A. Landman.  "3D whole brain segmentation using spatially localized atlas network tiles" 
NeuroImage 2019. 

Yuankai Huo, Zhoubing Xu, Katherine Aboud, Parasanna Parvathaneni, Shunxing Bao, Camilo Bermudez, Susan M. Resnick, Laurie E. Cutting, and Bennett A. Landman.  "Spatially Localized Atlas Network Tiles Enables 3D Whole Brain Segmentation" 
In International Conference on Medical Image Computing and Computer-Assisted Intervention, MICCAI 2018. 

```diff
+ The code and docker are free for noncommercial purposes.
+ The licence.md shows the terms for commercial and for-profit purposes.
```

## Quick Start
#### Get our docker image
```
sudo docker pull vuiiscci/slant:deep_brain_seg_v1_0_0
```
#### Run SLANT brain segmentation
You can run the following command or change the "input_dir", then you will have the final segmentation results in output_dir
```
# you need to specify the input directory
export input_dir=/home/input_dir   
# make that directory
sudo mkdir $input_dir
# download the test volume file, you can even put multiple input files here, no worries.
sudo wget -O  $input_dir/test_volume.nii.gz  https://www.nitrc.org/frs/download.php/10666/test_volume.nii.gz
# set output directory
export output_dir=$input_dir/output
#run the docker
sudo nvidia-docker run -it --rm -v $input_dir:/INPUTS/ -v $output_dir:/OUTPUTS masidocker/spiders:deep_brain_seg_v1_0_0 /extra/run_deep_brain_seg.sh
```
- You will see the final a segmentation file in "FinalResult"
- You will see the final a overlay pdf in "FinalPDF"
- You will see the final a txt file contains all label names and volume in "FinalVolTxt".

## Source Code
The SLANT is a whole brain segmentation pipeline that contains (1) pre-processing, (2) deep learning, (3) post-processing, which have all been contained in the Docker. The main scratch in Docker is the "run_deep_brain_seg.sh". The related source code and binary files have been included in the Docker. They can also be found in the "matlab" and "python".

- Pre- and Post-processing code can be found in "matlab"
- Train and testing code for deep learning part can be found in "python"

### If you only have CPU, you can use the following CPU version of the docker. (50GB Memory might be required using CPU!)
```
sudo docker pull vuiiscci/slant:deep_brain_seg_v1_0_0_CPU
```

## List of 45 Training Data and 5 Validation Data from OASIS study
### 45 Training
OAS1_0061_MR1|OAS1_0080_MR1|OAS1_0092_MR1|OAS1_0101_MR1|OAS1_0117_MR1|OAS1_0145_MR1|OAS1_0150_MR1|OAS1_0156_MR1|OAS1_0191_MR1|OAS1_0202_MR1|OAS1_0230_MR1|OAS1_0236_MR1|OAS1_0239_MR1|OAS1_0249_MR1|OAS1_0285_MR1|OAS1_0368_MR1|OAS1_0395_MR1|OAS1_0091_MR1|OAS1_0005_MR1|OAS1_0340_MR1|OAS1_0417_MR1|OAS1_0069_MR1|OAS1_0173_MR1|OAS1_0040_MR1|OAS1_0088_MR1|OAS1_0074_MR1|OAS1_0282_MR1|OAS1_0050_MR1|OAS1_0443_MR1|OAS1_0331_MR1|OAS1_0389_MR1|OAS1_0274_MR1|OAS1_0456_MR1|OAS1_0070_MR1|OAS1_0358_MR1|OAS1_0300_MR1|OAS1_0124_MR1|OAS1_0220_MR1|OAS1_0263_MR1|OAS1_0013_MR1|OAS1_0113_MR1|OAS1_0317_MR1|OAS1_0083_MR1|OAS1_0270_MR1|OAS1_0278_MR1
### 5 Validation
OAS1_0111_MR1|OAS1_0353_MR2|OAS1_0032_MR1|OAS1_0379_MR1|OAS1_0255_MR1
### You can download the list [HERE](https://github.com/MASILab/SLANTbrainSeg/blob/master/SLANT_neuroimage_train_test_split.xlsx)


## Detailed envrioment setting  

#### Testing platform
- Ubuntu 16.04
- cuda 8.0
- Pytorch 0.2
- Docker version 1.13.1-cs9
- Nvidia-docker version 1.0.1 to 2.0.3


#### install Docker
```
sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install docker-ce
```

#### install Nvidia-Docker
```
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```

## Detailed Results for Each ROI
The detailed measurement of Figure 9 in [NeuroImage](https://www.sciencedirect.com/science/article/pii/S1053811919302307) paper

Joint Label Fusion Multi Atlas: [Download CSV](https://github.com/MASILab/SLANTbrainSeg/blob/master/screenshot/Figure9_JLF_MAS.csv)
Non Local Spatial STAPLE Multi Atlas: [Download CSV](https://github.com/MASILab/SLANTbrainSeg/blob/master/screenshot/Figure9_NLSS_MAS.csv)
Patch CNN: [Download CSV](https://github.com/MASILab/SLANTbrainSeg/blob/master/screenshot/Figure9_Patch_CNN.csv)
Registration + U-Net: [Download CSV](https://github.com/MASILab/SLANTbrainSeg/blob/master/screenshot/Figure9_Reg+UNet.csv)
SLANT 8: [Download CSV](https://github.com/MASILab/SLANTbrainSeg/blob/master/screenshot/Figure9_SLANT_8.csv)
SLANT 27: [Download CSV](https://github.com/MASILab/SLANTbrainSeg/blob/master/screenshot/Figure9_SLANT_27.csv)

