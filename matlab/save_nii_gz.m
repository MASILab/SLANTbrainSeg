function save_nii_gz(nii, fileprefix, varargin)
% SAVE_NII_GZ - Saves a *.nii.gz file in matlab.
%
% There are two different forms of this function:
%
% 1 - save_nii_gz(nii, filenameIN)
% 2 - save_nii_gz(nii, filenameIN, filesuffix)
%
% Input: nii - the nifti struct
%        filenameIN - the .nii.gz file to load
%        filesuffix (opt) - an explicit filename suffix to add to the
%                           temporary file that is saved. This may be necessary
%                           if loading multiple files simultaneously as the
%                           default suffix is a pseudo-random number.
%

% set the temporary filename
if length(varargin) == 0
    pid = int32(feature('getpid'));
    randnum = num2str(round(rand()*1e8));
    filename = sprintf('/tmp/tmp-%d-%s.nii', pid, randnum);
else
    filename = ['/tmp/tmp', varargin{1}, '.nii'];
end

% try to save the temporary file
try
    save_nii(nii,filename);
catch
    disp(['Saving nii.gz file "untouched"']);
    disp(['The resulting nii file CANNOT be trusted']);
    save_untouch_nii(nii, filename);
end

% zip the temporary file to its final location
system(sprintf('gzip -c %s > %s',filename,fileprefix));

% delete the temporary file
delete(filename);

