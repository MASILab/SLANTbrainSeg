# SLANT_brain_seg
Deep Whole Brain Segmentation Using SLANT Method. 

A T1 MRI scan can be segmented to 133 labels based on BrainCOLOR protocal(http://braincolor.mindboggle.info/protocols/).

It has been implemented as a single Docker.
```diff
- Please cite the following MICCAI 2018 paper [SLANT](https://arxiv.org/pdf/1806.00546.pdf), if you used the SLANT whole brain segmentation.
```
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


