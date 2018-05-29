clc;clear;close all;

% test on local machine
orig_dir = '/share4/xiongy2/McHugo/input_test';
output_dir = '/share4/xiongy2/McHug/output_test'; 

% run on docker
% orig_dir = '/INPUTS';
% output_dir = '/OUTPUTS/'; 

disp('Start Making PDF...');
% try
    % run the pipeline
    makePDFviews(orig_dir,output_dir);
% catch e
%     getReport(e)
% end
disp('Finish Making PDF...');
