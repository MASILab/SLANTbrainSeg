function fusion_Seg_file = get_MV_fused_image(working_root_dir, MNI_seg_dir, subName, epoch_str, pieces, work_key)

fusion_Seg_file = [MNI_seg_dir filesep sprintf('%s_seg.nii.gz',subName)];


if ~exist(fusion_Seg_file)
    fusion_4D = uint8(zeros(172,220,156,length(pieces)));
    final_seg = uint8(zeros(172,220,156));
    for pi = 1:length(pieces)
        piece = pieces{pi};
        working_dir = [working_root_dir filesep piece filesep work_key filesep 'seg_output'];
        final_output_root_dir  = [working_dir filesep epoch_str filesep 'seg_orig_final'];
        final_Seg_piece = [final_output_root_dir filesep sprintf('%s_seg.nii.gz',subName)];
        %         valid_region = fusion_map(strcmp(fusion_map(:,1), piece),2:end);
        nii = load_untouch_nii_gz(final_Seg_piece);
        fusion_4D(:,:,:,pi) = nii.img;
        
    end
    
    
    
    for i = 1:size(fusion_4D,1)
        for j = 1:size(fusion_4D,2)
            for k = 1:size(fusion_4D,3)
                voxels = squeeze(fusion_4D(i,j,k,:));
                final_seg(i,j,k) = mode(voxels(voxels~=255));
            end
        end
    end
    
    
    nii.img = final_seg;
    save_untouch_nii_gz(nii,fusion_Seg_file);
    
end


end