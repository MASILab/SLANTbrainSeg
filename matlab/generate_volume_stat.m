clc;clear;close all;

% addpath(genpath('/fs4/masi/huoy1/FS3_backup/software/full-multi-atlas/masi-fusion/src'));
% addpath(genpath('/share4/huoy1/Deep_5000_Brain/code/evaluation'));

% test on local machine
% mdir = '/share4/xiongy2/McHug/output_test';
% final_out_dir = '/share4/xiongy2/McHug/output_test/FinalVolTxt'; 
% target_dir = '/share4/xiongy2/McHugo/input_test';
% extra_dir = '/share4/xiongy2/docker/extra';

% run on docker 
mdir = '/OUTPUTS';
final_out_dir = '/OUTPUTS/FinalVolTxt'; 
target_dir = '/INPUTS';
extra_dir = '/extra';

sublist = dir([target_dir filesep '*.nii.gz']);

if ~isdir(final_out_dir);mkdir(final_out_dir);end;

for si = 1:length(sublist)
    subFile = sublist(si).name;
    subName = get_basename(subFile);
    target_fname = [target_dir filesep subFile];
    res_orig_seg_fname = [mdir filesep 'FinalResult' filesep sprintf('%s_seg.nii.gz',subName)];
   res_txt_fname = [final_out_dir filesep sprintf('%s_label_volumes.txt',subName)];
   
    if ~exist(res_txt_fname)
        save_txt_file( res_txt_fname, res_orig_seg_fname)
    end
end
