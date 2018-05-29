function age = get_file_age(fname)
% GET_FILE_AGE - get the age of a specified file
%                (i.e., how long ago it was last modified)
%
% age = get_file_age(fname)
%
% Input: fname - the filename to check
%
% Output: age - the age of the file (in seconds)

if ~exist(fname, 'file')
    age = -1;
    return;
end

% get the age of the file
[status,age] = unix(['echo $(( $(date +%s) - $(stat -c %Y ', fname, ')))']);

age = str2num(age);

