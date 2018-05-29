function preproc_pipline(target_fname, mdir, in)
% MULTI_ATLAS_PIPELINE - runs the complete multi-atlas pipeline for large-scale
%                        brain analysis
%
% multi_atlas_pipeline(target_fname, mdir, in);
%
% Input: target_fname - the target image filename
%        mdir - the output directory where all preliminary results will be saved
%        in - the input options struct (see below)
%
% Output: None, however, several files are created
%
% -- The Input Options ("in") --
%
% * Mandatory Options *
% in.niftyreg_loc - the directory containing the niftyreg binaries
% in.ants_loc - the directory containing the ants install
% in.atlas_loc - the atlas processing directory
% in.mni_loc - the directory containing the MNI template
% in.mipav_loc - the mipav install directory
%
% * Non-Mandatory Options *
% in.runtype - the method to run the approach, either 'single' or 'cluster'
%                 - Note: 'cluster' assumes an SGE cluster (i.e., not ACCRE)
%                 - default = 'single'
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Error Check the Input Settings
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~isfield(in, 'niftyreg_loc')
    error('ERROR: in.niftyreg_loc not specified');
end

if ~isfield(in, 'atlas_loc')
    error('ERROR: in.atlas_loc not specified');
end
if ~isfield(in, 'mni_loc')
    error('ERROR: in.mni_loc not specified');
end

if ~isfield(in, 'in.runtype')
    in.runtype = 'single';
end
if ~strcmp(in.runtype, 'single') && ~strcmp(in.runtype, 'cluster')
    error('ERROR: Invalid in.runtype -- should be ''single'' or ''cluster''');
end
if ~exist(in.niftyreg_loc, 'dir')
    error('ERROR: in.niftyreg_loc does not exist');
end
if ~exist(in.ants_loc, 'dir')
    error('ERROR: in.ants_loc does not exist');
end
if ~exist(in.atlas_loc, 'dir')
    error('ERROR: in.atlas_loc does not exist');
end
if ~exist(in.mni_loc, 'dir')
    error('ERROR: in.mni_loc does not exist');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Hidden Settings
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% MNI Settings
opts.MNI.fname = [in.mni_loc, 'average305_t1_tal_lin.nii.gz'];
opts.MNI.msk_fname = [in.mni_loc, 'average305_t1_tal_lin_mask.nii.gz'];
opts.MNI.loc = in.niftyreg_loc;


% Atlas Settings
atlas_model_fname = [in.atlas_loc, 'atlas_model.mat'];
atlas_hierarchy_fname = [in.atlas_loc, 'braincolor_hierarchy_STAPLE.txt'];

% N4 Correction Settings
N4loc = [in.ants_loc, 'bin/N4BiasFieldCorrection'];
opts.N4.biasfield = true;


% Multi-Atlas Registration Options
regtype = 'aladin-ants';
regloc = ['. ~/.bashrc && ', in.ants_loc];
opts.reg = struct;
opts.reg.memval = {'3G', '5G'};
opts.reg.aladin_loc = in.niftyreg_loc;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Run all of the analysis
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% start the timer for keeping track of runtime
tic;

% make the output directories if necessary
if ~exist(mdir, 'dir'), mkdir(mdir); end
target_bname = get_basename(target_fname);
out_dir = [mdir, '/', target_bname, '/'];
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
tmp_dir = [out_dir, '/temp-out/'];
if ~exist(tmp_dir, 'dir'), mkdir(tmp_dir); end

tprintf('%s\n', repmat('*', [1 60]));
tprintf('Target Image: %s\n', target_fname);
tprintf('Output Directory: %s\n', out_dir);
tprintf('NiftyReg Location: %s\n', in.niftyreg_loc);
tprintf('ANTs Location: %s\n', in.ants_loc);
tprintf('Atlas Processing Location: %s\n', in.atlas_loc);
tprintf('MNI Template Location: %s\n', in.mni_loc);
tprintf('Run Type: %s\n', in.runtype);
tprintf('%s\n', repmat('*', [1 60]));
tprintf('\n');

tprintf('*** Running Pre-Processing ***\n');
tprintf('\n');

% set the output filenames
res_norm_fname = sprintf('%starget_processed.nii.gz', out_dir);
res_norm_seg_fname = sprintf('%starget_processed_seg.nii.gz', out_dir);
res_orig_fname = sprintf('%sorig_target.nii.gz', out_dir);
res_orig_seg_fname = sprintf('%sorig_target_seg.nii.gz', out_dir);
res_txt_fname = sprintf('%starget_processed_label_volumes.txt', out_dir);
res_pdf_fname = sprintf('%starget_processed_summary.pdf', out_dir);
res_zip_fname = sprintf('%starget_processed_output.zip', out_dir);

% copy the original target to the output directory
if ~exist(res_orig_fname)
    system(sprintf('cp %s %s\n', target_fname, res_orig_fname));
end

% Rigidly align the target image with the MNI305 image
[res_fname aff_fname inv_fname] = rigidMNI(target_fname, out_dir, opts.MNI);

% Run inhomogeneity correction
res_corr_fname = run_N4correction(res_fname, out_dir, N4loc, ...
                                  in.runtype, opts.N4);
res_corr_fname = res_corr_fname{1};

% Do the remaining pre-processing
[ll s_inds] = process_target(atlas_model_fname, res_corr_fname, res_norm_fname);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% End of Main Function
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end



% function to register target to MNI space
function [res_fname aff_fname inv_fname] = rigidMNI(target_fname, out_dir, opts)

    % set some directories / filenames
    reg_dir = [out_dir, 'MNI-registration/'];
    if ~exist(reg_dir, 'dir'), mkdir(reg_dir); end
    tmp_dir = [out_dir, '/temp-out/'];
    if ~exist(tmp_dir, 'dir'), mkdir(tmp_dir); end
    txt_out = [tmp_dir, 'MNI-registration.txt'];
    res_fname = sprintf('%starget_MNI.nii.gz', reg_dir);
    aff_fname = sprintf('%starget_MNI_aff.mtx', reg_dir);
    inv_fname = sprintf('%starget_MNI_aff_inv.mtx', reg_dir);

    tprintf('-> Registering target image to MNI space\n');
    tprintf('Target Image: %s\n', target_fname);
    tprintf('MNI Image: %s\n', opts.fname);
    tprintf('MNI Mask: %s\n', opts.msk_fname);

    % skip the registration if we can
    if ~exist(res_fname) || ~exist(aff_fname)

        % run the registration
        aladin_cmd = sprintf(['%s/reg_aladin -ref %s -flo %s -aff %s ', ...
                              '-res %s -rmask %s >> %s 2>&1\n'], ...
                              opts.loc, opts.fname, target_fname, ...
                              aff_fname, res_fname, opts.msk_fname, txt_out);
        system(aladin_cmd);


        % try and fix any obvious problems with the nifty file
        target_nii = load_untouch_nii_gz(res_fname);
        target_nii.img = single(target_nii.img) * ...
                         target_nii.hdr.dime.scl_slope + ...
                         target_nii.hdr.dime.scl_inter;
        target_nii.hdr.dime.datatype = 16;
        target_nii.hdr.dime.bitpix = 32;
        target_nii.hdr.dime.scl_slope = 1;
        target_nii.hdr.dime.scl_inter = 0;
        save_untouch_nii_gz(target_nii, res_fname);
        tprintf('Registration finished (see %s)\n', txt_out);
    else
        tprintf('Skipping registration (output files exist)\n');
    end

    tprintf('Computing inverse affine transformation\n');
    if ~exist(inv_fname)
        inv_cmd = sprintf('%s/reg_transform -ref %s -invAff %s %s\n', ...
                          opts.loc, res_fname, aff_fname, inv_fname);
        system(inv_cmd);
    else
        tprintf('Skipping inverse affine calculation (output files exist)\n');
    end

    tprintf('Rigid Registration result: %s\n', res_fname);
    tprintf('Rigid Registration matrix: %s\n', aff_fname);
    tprintf('Inverse Rigid Registration matrix: %s\n', aff_fname);
    tprintf('\n');
end

% function for projecting registered target to atlas space
function [likelihood s_inds] = process_target(atlas_model_fname, ...
                                              res_corr_fname, ...
                                              res_norm_fname)

    tprintf('-> Loading Multi-Atlas Appearance model: %s\n', atlas_model_fname);
    load(atlas_model_fname, 'model_seg', 'brain_seg', 'meanvec', 'eigenvecs',...
                            'eigenvals', 'weights', 'percvar', 'model_sl', ...
                            'model_sls', 'norm_regressand', 'dims', 'datamat');
    tprintf('\n');

    tprintf('-> Loading Processed Reclean_sublist(sublist);gistration Result\n');
    target_nii = load_untouch_nii_gz(res_corr_fname);
    tprintf('\n');

    tprintf('-> Normaling target intensities to model space\n');
    target_nii.img = double(target_nii.img);
    meanval = mean(target_nii.img(brain_seg > 0));
    stdval = std(target_nii.img(brain_seg > 0));
    target_nii.img = (target_nii.img - meanval) / stdval;
    targetvec = sort(target_nii.img(brain_seg>0));

    % regress the target to the mean atlas image
    betas = robustfit(targetvec, norm_regressand, 'huber');
    tprintf('Found the robust (huber) regression parameters: %f %f\n', ...
            betas(1), betas(2));

    % apply the regression parameters
    target_nii.img = single(target_nii.img * betas(2) + betas(1));

    % save the normalized image
    target_nii.hdr.dime.glmax = max(target_nii.img(:));
    target_nii.hdr.dime.glmin = min(target_nii.img(:));
    save_untouch_nii_gz(target_nii, res_norm_fname);
    tprintf('Saved normalized image to: %s\n', res_norm_fname);
    tprintf('\n');

    tprintf('-> Projecting target to model space\n');
    targetvec = double(target_nii.img(model_seg > 0));

    % project the target into the manifold
    proj_weights = eigenvecs' * (targetvec - meanvec);

    % get the difference between the projected weights and the modeled atlases
    diffs = weights - repmat(proj_weights', [size(weights, 1) 1]);
    ww = sqrt(sum(diffs.^2, 2));
    [s_diffs s_inds] = sort(ww);

    recon_targetvec = meanvec + eigenvecs*proj_weights;
    likelihood = exp(corr(targetvec, recon_targetvec) - 1);
    tprintf('Estimated likelihood: %f\n', likelihood);
    tprintf('\n');
end


