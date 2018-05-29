clc;clear;close all;

addpath(genpath('/fs4/masi/huoy1/FS3_backup/software/full-multi-atlas/masi-fusion/src'))

input_root_path = '/share4/xiongy2/rawdata/FirstHalf';
output_root_path = '/share4/xiongy2/QA';
D = dir(input_root_path);
for k = 180:length(D)
    subName = D(k).name
    currImg = [input_root_path filesep subName];
    files = dir([currImg filesep '*.nii.gz']);
    s = numel(files);
    if s <= 4
        continue
    end
    for file = files'
        filename = [currImg filesep file(1).name];
        outputfilename = strrep(file(1).name,'.nii.gz','');
        outputname = [output_root_path filesep sprintf('%s.tif',outputfilename)];
        if ~exist(outputname)
            a = load_untouch_nii_gz(filename); % load the nii.gz file
            figure;
            count = 1;
            d = [];
            for slice = 1:5:size(a.img,3)
                b = a.img(:,:,slice);
                c = ind2rgb(b,gray(256));
%                 c = cat(3,b,b,b);
                d(:,:,:,count) = c;
                count = count+1;
            end;
            montage(d);
            print(gcf,'-r 300','-dtiff',outputname);
            close(gcf);
        end
    end;
end;