LOCAL_INPUTS_PATH=$(readlink -f ${1-/INPUTS}) #/INPUTS
LOCAL_OUTPUTS_PATH=$(readlink -f ${2-/OUTPUTS}) # /OUTPUTS
LCOAL_CONFIG_PATH=$(readlink -f ${3-/config.yaml}) # /config.yaml

#change local input output path to be compatible with slant
# you need to specify the input directory
export input_dir=$LOCAL_INPUTS_PATH/NIFTI
# set output directory
export output_dir=$LOCAL_OUTPUTS_PATH/temp
#run the docker
sudo docker run -it --rm -v $input_dir:/INPUTS/ -v $output_dir:/OUTPUTS masidocker/public:deep_brain_seg_v1_1_0_CPU /extra/run_deep_brain_seg.sh

#orgnize your outcomes based on the required format of the model zoo
mkdir $LOCAL_OUTPUTS_PATH/PDF
cp $output_dir/FinalPDF/*.pdf $LOCAL_OUTPUTS_PATH/PDF

mkdir $LOCAL_OUTPUTS_PATH/Segmentation
mkdir $LOCAL_OUTPUTS_PATH/Segmentation/NIFTI
cp $output_dir/FinalResult/*.nii.gz $LOCAL_OUTPUTS_PATH/Segmentation/NIFTI

mkdir $LOCAL_OUTPUTS_PATH/Metrics
mkdir $LOCAL_OUTPUTS_PATH/Metrics/CSV
cp $output_dir/FinalVolTxt/*.txt $LOCAL_OUTPUTS_PATH/Metrics/CSV
for f in $LOCAL_OUTPUTS_PATH/Metrics/CSV/*.txt; do
    mv "$f" "${f%.txt}.csv"
done
