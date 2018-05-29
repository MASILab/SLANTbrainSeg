clc;clear;close all;

% run on docker
% mdir = '/OUTPUTS'; 
% target_dir = '/INPUTS'; 
% final_out_dir = '/OUTPUTS/FinalResult';
% extra_dir = '/extra';

% test on local machine
mdir = '/share4/xiongy2/docker/OUTPUTS';
final_out_dir = '/share4/xiongy2/docker/OUTPUTS/FinalResult'; 
target_dir = '/share4/xiongy2/docker/INPUTS';
extra_dir = '/share4/xiongy2/docker/extra';

atlas_label_fname = [extra_dir filesep  'full-multi-atlas' filesep 'atlas-processing' filesep 'atlas-label-info.csv'];


sublist = dir([target_dir filesep '*.nii.gz']);

for si = 1:length(sublist)
    subFile = sublist(si).name;
    subName = get_basename(subFile);
    target_fname = [target_dir filesep subFile];
    
    output_final = [final_out_dir filesep sprintf('%s_seg.nii.gz',subName)];
    res_txt_fname = [final_out_dir filesep sprintf('%s_label_volumes.txt',subName)];
    res_pdf_fname = [final_out_dir filesep sprintf('%s_QA.pdf',subName)];
    
    normed_dir = [mdir filesep subName];
    working_dir = [normed_dir filesep 'working_dir'];
    res_norm_fname = [normed_dir filesep 'target_processed.nii.gz'];
    res_norm_seg_fname = [working_dir filesep 'all_piece' filesep '5000_fusion27_mv' filesep ...
        'test_out' filesep 'seg_output' filesep 'epoch_0005' filesep 'seg_orig_final' filesep 'target_processed_seg.nii.gz'];
    tmp_dir = [normed_dir filesep 'temp-out' filesep];
    
    % save the text file
    fprintf('-> Saving the Summary TXT\n');
    fprintf('TXT File: %s\n', res_txt_fname);
    save_txt_file(atlas_label_fname, res_txt_fname, output_final);
    
    % save the pdf summary
    fprintf('-> Saving the Summary PDF\n');
    fprintf('PDF File: %s\n', res_pdf_fname);
    [proj_name, subj_name, expr_name] = get_XNAT_info(target_fname);
    generate_braincolor_pdf(res_norm_fname, res_norm_seg_fname, res_pdf_fname, ...
        tmp_dir, proj_name, subj_name, expr_name);
end

