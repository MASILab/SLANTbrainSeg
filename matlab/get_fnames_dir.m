function fnames = get_fnames_dir(dirname, sfix, varargin)
% get_fnames_dir - get a cell array of filenames from a directory and a
%                  file regular expression
%
% Two forms:
% 1) fnames = get_fnames_dir(dirname, sfix)
% 2) fnames = get_fnames_dir(dirname, sfix, use_recursive)
%
%
% Input: dirname - the directory location
%        sfix - the file regular expression
%             - e.g., '*.nii.gz', '*.txt'
%        use_recursive - (optional) recursively search through directories
%                      - boolean: true/false
%
% Output: fnames - cell array of full path filenames
%

% make sure the input is a directory
if ~exist(dirname, 'dir')
    error('Input "dirname" is not a directory: %s', dirname);
end

% add a file separator if we have to
if ~strcmp(dirname(end), filesep)
    dirname = [dirname, filesep];
end

use_recursive = false;

if length(varargin) == 1
    if varargin{1}
        use_recursive = true;
    end
elseif length(varargin) > 1
    error('too many input arguments');
end

cc = 1;

fnames = cell(0);

if ~use_recursive

    % get all of the filenames
    if isdir([dirname, sfix])
        tmpnames(1).name = sfix;
    else
        tmpnames = dir([dirname, sfix]);
    end

    % set the number of files
    num_files = length(tmpnames);

    % set the fullpath filenames
    for i = 1:num_files
        if ~strcmp(tmpnames(i).name, '.') && ~strcmp(tmpnames(i).name, '..')
            fnames{cc} = fullfile(dirname, tmpnames(i).name);
            cc = cc + 1;
        end
    end

else
    % get the matching filenames
    match_fnames = get_fnames_dir(dirname, sfix);

    % get the remaining filenames
    all_fnames = setdiff(get_fnames_dir(dirname, '*'), match_fnames);

    % recursively search all sub-directories
    for i = 1:length(all_fnames)
        if isdir(all_fnames{i})
            fnames = [fnames, get_fnames_dir(all_fnames{i}, sfix, true)];
        end
    end

    % add the matching filenames
    fnames = [fnames, match_fnames];
end

