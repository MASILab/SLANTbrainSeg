# SLANT_brain_seg
Deep Whole Brain Segmentation Using SLANT Method
Docker version 1.13.1-cs9
Nvidia-docker version 1.0.1 / 2.0.3
sigularity 2.4.2


#install Docker
sudo apt-get install apt-transport-https ca-certificates curl software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

sudo apt-get update

sudo apt-get install docker-ce

#install singularity
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \  
  sudo apt-key add -

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update

sudo apt-get install -y nvidia-docker2



