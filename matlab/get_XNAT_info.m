function [proj_name, subj_name, expr_name] = get_XNAT_info(fname)
% get_XNAT_info - get the meta-information from an XNAT filename
%
% [proj_name, subj_name, expr_name] = get_XNAT_info(fname)
%
% Input: fname - the input filename
%
% Output: proj_name - the project name
%         subj_name - the subject name
%         expr_name - the experiment name
%

bname = get_basename(fname);
field_sep_locs = strfind(bname,'-x-');
if length(field_sep_locs) == 3
    proj_name = bname(1:field_sep_locs(1)-1);
    subj_name = bname(field_sep_locs(1)+3:field_sep_locs(2)-1);
    expr_name = bname(field_sep_locs(2)+3:field_sep_locs(3)-1);
else
    proj_name = 'Unknown';
    subj_name = 'Unknown';
    expr_name = 'Unknown';
end
