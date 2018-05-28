# SLANT_brain_seg
Deep Whole Brain Segmentation Using SLANT Method. 
A T1 MRI scan can be segmented to 133 labels based on BrainCOLOR protocal(http://braincolor.mindboggle.info/protocols/).
It has been implemented as a single Docker.
```diff
- Please cite the MICCAI 2018 paper if you used the SLANT whole brain segmentation.
```
```diff
+ Yuankai Huo, Zhoubing Xu, Katherine Aboud, Parasanna Parvathaneni, Shunxing Bao, 
Camilo Bermudez, Susan M. Resnick, Laurie E. Cutting, and Bennett A. Landman. 
"Spatially Localized Atlas Network Tiles Enables 3D Whole Brain Segmentation" 
In International Conference on Medical Image Computing and Computer-Assisted Intervention, MICCAI 2018. 
```

## Quick Start
#### get our docker image
```
sudo docker pull masidocker/spiders:deep_brain_seg_v1_0_0
```
#### run SLANT brain segmentation
```
export input_dir=/home/input_dir   #you need to specify the directory
wget https://www.nitrc.org/frs/download.php/10666/test_volume.nii.gz

sudo nvidia-docker run -it --rm -v /home/yuankai/input_dir/:/INPUTS/ -v /home/yuankai/output_dir:/OUTPUTS masidocker/spiders:deep_brain_seg_v1_0_0 /extra/run_deep_brain_seg.sh
```

# Detailed envrioment setting  

## Testing platform
Ubuntu 16.04
cuda 8.0
Pytorch 0.2
Docker version 1.13.1-cs9
Nvidia-docker version 1.0.1 to 2.0.3


## install Docker
```
sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install docker-ce
```

## install Nvidia-Docker
```
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```


