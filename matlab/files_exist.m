function val = files_exist(files)
% FILES_EXIST - checks whether or not a cell array of files exist
%
% val = files_exist(files)
%
% Input: files - a cell array of filenames
%
% Output: val - all exist (1) or not (0)

val = 1;
for i = 1:length(files)
    if get_file_age(files{i}) < 20
        val = 0;
    end
end
