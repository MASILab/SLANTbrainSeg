function nii = load_nii_gz(filenameIN, varargin)
% LOAD_NII_GZ - Loads a *.nii.gz file in matlab.
%
% There are two different forms of this function:
%
% 1 - nii = load_nii_gz(filenameIN)
% 2 - nii = load_nii_gz(filenameIN, filesuffix)
%
% Input: filenameIN - the .nii.gz file to load
%        filesuffix (opt) - an explicit filename suffix to add to the
%                           temporary file that is saved. This may be necessary
%                           if loading multiple files simultaneously as the
%                           default suffix is a pseudo-random number.
%
% Output: nii - The nifti struct.

useprefix = 0;
if length(varargin) == 1
    if length(varargin{1} > 0)
        useprefix = 1;
    end
end

% set the temporary filename
if ~useprefix
    pid = int32(feature('getpid'));
    randnum = num2str(round(rand()*1e8));
    filename = sprintf('/tmp/tmp-%d-%s.nii', pid, randnum);
else
    filename = ['/tmp/tmp', varargin{1}, '.nii'];
end

% unzip to the temporary filename
system(sprintf('gzip -dc %s > %s',filenameIN,filename));

% try to load the nifti file
try
    nii = load_nii(filename);
catch
    disp(['Something is strange. Loading it untouched: ' filenameIN])
    disp(['The resulting nii file CANNOT be trusted']);
    nii = load_untouch_nii(filename);
    nii.img = flipdim(nii.img, 2);
end

% delete the temporary file
delete(filename);
