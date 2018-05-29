function orig_inv_seg_file = postproc_pipeline(target_fname,working_dir,in)


testing_T1_root_dir = [working_dir filesep 'deep_learning'];
working_root_dir = [working_dir filesep 'all_piece'];


% method = 'cluster';
method = 'single';

% run_orig_reg_method = 'cluster';
run_orig_reg_method = 'single';

tic;
[piece_map, fusion_map] = get_piece_map();


sublist = dir([testing_T1_root_dir filesep '*.nii.gz']);

stage = 'firstroundfusion'; % 27 + fusion


if  strcmp(stage,'firstroundfusion')
    pieces = {'1_1_1','2_1_1','3_1_1','1_2_1','2_2_1','3_2_1','1_3_1','2_3_1',...
        '3_3_1','1_1_2','2_1_2','3_1_2','1_2_2','2_2_2','3_2_2','1_3_2','2_3_2',...
        '3_3_2','1_1_3','2_1_3','3_1_3','1_2_3','2_2_3','3_2_3','1_3_3','2_3_3','3_3_3'};
    out_key = 'test_out';
    work_key = 'testing';
    epochs = [27];
    fusion_name = '5000_fusion27_mv';
end


if strcmp(stage,'naive') || strcmp(stage,'semisuper') || strcmp(stage,'firstroundfusion') || strcmp(stage,'secondroundfinetune')
    for pi = 1:length(pieces)
        piece = pieces{pi};
        
        working_dir = [working_root_dir filesep piece filesep work_key filesep 'seg_output'];
        deep_seg_dir = [working_root_dir filesep piece filesep out_key filesep 'seg_output'];
        
        working_region = piece_map(strcmp(piece_map(:,1), piece),2:end);        
        
        for ei = 1:length(epochs)
            
            epoch = epochs(ei);
            epoch_str = sprintf('epoch_%04d',epoch);
            testing_seg_root_data           = [deep_seg_dir filesep epoch_str filesep 'seg'];
            testing_seg_resample_root_data  = [working_dir filesep epoch_str filesep 'seg_resample'];
            MNI_root_dir                    = [working_dir filesep epoch_str filesep 'seg_MNI'];
            output_root_dir                 = [working_dir filesep epoch_str filesep 'seg_orig'];
            final_output_root_dir           = [working_dir filesep epoch_str filesep 'seg_orig_final'];
            dice_mat_dir                    = [working_dir filesep epoch_str filesep 'dice_mat'];
            
            for si = 1:length(sublist)
                subName = get_basename(sublist(si).name);
                
                T1_resample = [testing_T1_root_dir filesep sprintf('%s.nii.gz',subName)];                
                
                if ~isdir(testing_seg_resample_root_data);mkdir(testing_seg_resample_root_data);end;
                Seg_raw = [testing_seg_root_data filesep sprintf('%s_seg.nii.gz',subName)];
                Seg_resample = [testing_seg_resample_root_data filesep sprintf('%s_seg.nii.gz',subName)];
                
                project_space(T1_resample,Seg_raw,Seg_resample,working_region)
                
                %project the segmentation file to original MNI space
                final_Seg = [final_output_root_dir filesep sprintf('%s_seg.nii.gz',subName)];
                reassign_braincolor(Seg_resample,final_Seg);
                

            end
        end
    end
end


%merge seg
for si = 1:length(sublist)
    subName = get_basename(sublist(si).name);

    
    for ei = 1:length(epochs)
        epoch = epochs(ei);
        epoch_str = sprintf('epoch_%04d',epoch);
        fprintf('start label fusion si=%d, ei=%d\n',si,ei);
        toc;
        fusion_root_dir = [working_root_dir filesep fusion_name filesep out_key filesep 'seg_output'];
        MNI_seg_dir = [working_root_dir filesep fusion_name filesep out_key filesep 'seg_output' filesep epoch_str filesep 'seg_orig_final'];
        if ~isdir(MNI_seg_dir);mkdir(MNI_seg_dir);end;
        
        if strcmp(fusion_name,'5000_fusion27_mv')
            if strcmp(method,'cluster')
                get_MV_fused_image_cluster(working_root_dir, MNI_seg_dir, subName, epoch_str, pieces, work_key);
                continue;
            else
                final_Seg = get_MV_fused_image(working_root_dir, MNI_seg_dir, subName, epoch_str, pieces, work_key);
            end
        elseif strcmp(fusion_name,'5000_fusion8')
            final_Seg = get_fused_image(working_root_dir, MNI_seg_dir, subName, epoch_str, pieces, fusion_map, work_key);
        end
        
        toc;
        fprintf('start inv reg si=%d, ei=%d\n',si,ei);
        
        %orig space
        T1_orig_file = target_fname;
        
        %MNI 
        T1_MNI = [testing_T1_root_dir filesep  sprintf('%s.nii.gz',subName)];
%         Manual_orig_seg = [orig_seg_root_dir filesep sprintf('%s_glm.nii.gz',subName)];
        
        dice_orig_mat_dir = [fusion_root_dir filesep epoch_str filesep 'dice_orig_mat'];
        if ~isdir(dice_orig_mat_dir);mkdir(dice_orig_mat_dir);end;
        
        orig_invreg_dir = [fusion_root_dir filesep epoch_str filesep 'inv_orig_space'];
        if ~isdir(orig_invreg_dir);mkdir(orig_invreg_dir);end;
        orig_inv_seg_file = [orig_invreg_dir filesep sprintf('%s_orig_seg.nii.gz',subName)];
        
        inv_reg_working_dir = [orig_invreg_dir filesep sprintf('%s_inv_reg',subName)];
        
        if ~exist(orig_inv_seg_file)
            cmd_aaa = apply_aladin_affine_via_ants( in.niftyreg_loc, ...
                [in.ants_loc, '/bin/'], ...
                orig_inv_seg_file, ...
                T1_MNI, ...
                final_Seg, ...
                T1_orig_file, ...
                [inv_reg_working_dir, '/inverse-MNI-registration/'], ...
                [inv_reg_working_dir, '/temp/']);
            
            if strcmp(run_orig_reg_method,'single')
                run_cmd_single(cmd_aaa);
            else
                cluster_dir = [inv_reg_working_dir filesep 'PBS_LOG' ];
                if ~exist(cluster_dir);mkdir(cluster_dir);end
                pbsfile = [cluster_dir filesep 'run_rev_reg.pbs'];
                txtfile = [cluster_dir filesep 'run_rev_reg.txt'];
                cd(cluster_dir);
%                 run_cmd_cluster(cmd_aaa, {'4G','6G'}, pbsfile, txtfile, 'hadoop_queue');
                run_cmd_cluster(cmd_aaa, {'3G','5G'}, pbsfile, txtfile, 'clusterjob');
                continue;
            end
        end
        toc;

        
        fprintf('fusion done si=%d, ei=%d\n',si,epoch);
        
    end
end

return;