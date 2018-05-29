function run_cmd_single(cmds, varargin)
% RUN_CMD_SINGLE - runs a collection of commands (cell array) on the local
%                  machine
%
% Two forms:
% 1) run_cmd_single(cmds)
% 2) run_cmd_single(cmds, txtout)
%
% Input: cmds - a cell array of commands
%        txtout - the text file where all output will be saved
% Output: (NONE)

if length(varargin) == 0
    for i = 1:length(cmds)
        system(cmds{i});
    end
elseif length(varargin) == 1
    for i = 1:length(cmds)
        cc = cmds{i};
        if strcmp(cc(end), sprintf('\n'))
            cc = cc(1:(end-1));
        end
        system(sprintf('%s >> %s 2>&1\n', cc, varargin{1}));
    end
else
    error('too many input arguments');
end

