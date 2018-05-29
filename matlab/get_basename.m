function basename = get_basename(fname)
% get_basename - gets the basename of a file (robust to double endings)
%                 - if fname = '/test/whatever.nii.gz'
%                 - then returns 'whatever'
%
% prefix = get_basename(fname)
%
% Input: fname - the input filename
%
% Output: basename - the file basename
%

% get the basename
[~, basename, ~] = fileparts(fname);
if strcmp(basename(end-3:end), '.nii')
    [~, basename, ~] = fileparts(basename);
end


