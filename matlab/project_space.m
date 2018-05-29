function project_space(orig_file,target_file,new_file,output_loc)

if nargin < 4
    output_loc = [];
end

if ~exist(new_file)
    if isempty(output_loc)
        orig_nii = load_untouch_nii_gz(orig_file);
        seg_resample_nii = load_untouch_nii_gz(target_file);
        seg_orig_nii = orig_nii;
        seg_orig_nii.hdr.dime.datatype = 2;
        seg_orig_nii.hdr.dime.bitpix = 8;
        seg_orig_nii.hdr.dime.glmax = 255;
        seg_orig_nii.hdr.dime.glmin = 0;
        seg_orig_nii.hdr.dime.scl_inter = 0;
        im=seg_resample_nii.img; %%% input image
        imOut = permute(im,[2,3,1]);
        seg_orig_nii.img = imOut;
        save_untouch_nii_gz(seg_orig_nii,new_file);
    else
        orig_nii = load_untouch_nii_gz(orig_file);
        seg_resample_nii = load_untouch_nii_gz(target_file);
        seg_orig_nii = orig_nii;
        seg_orig_nii.hdr.dime.datatype = 2;
        seg_orig_nii.hdr.dime.bitpix = 8;
        seg_orig_nii.hdr.dime.glmax = 255;
        seg_orig_nii.hdr.dime.glmin = 0;
        seg_orig_nii.hdr.dime.scl_inter = 0;
        im=seg_resample_nii.img; %%% input image
        imOut = permute(im,[2,3,1]);
        seg_orig_nii.img = ones(size(seg_orig_nii.img))*255;
        seg_orig_nii.img(output_loc{1}+1:output_loc{2},...
            output_loc{3}+1:output_loc{4},output_loc{5}+1:output_loc{6}) = imOut;
        save_untouch_nii_gz(seg_orig_nii,new_file);
    end
end

end