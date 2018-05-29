function [corr_targets] = run_N4correction(targets, out_dir, ...
                                           N4loc, runtype, varargin)
% RUN_N4CORRECTION - runs N4 correction on a collection targets
%
% corr_targets = run_N4correction(targets, out_dir, ...
%                                 N4loc, runtype, opts)
%
% Input: targets - a cell array of target filenames
%        out_dir - the output directory location
%        N4loc - the location of the N4 correction application
%        runtype - either 'cluster' or 'single'
%        opts - (optional) a struct of options
%
% Output: corr_targets - a cell array of corrected targets

% make sure the output directory exists
if ~exist(out_dir, 'dir')
    error(sprintf('Output directory %s does not exist', out_dir));
end

% if the targets is a single file, make it a cell array.
if ~iscell(targets)
    tt = targets;
    targets = cell(1);
    targets{1} = tt;
end

if length(varargin) == 0
    opts = struct;
else
    opts = varargin{1};
end

% set the default options
if ~isfield(opts, 'memval')
    opts.memval = '2G';
end
if ~isfield(opts, 'pbsfile')
    opts.pbsfile = sprintf('N4-pbsfile.pbs');
end
if ~isfield(opts, 'txtout')
    opts.txtout = sprintf('N4-txtout.txt');
end
if ~isfield(opts, 'addopts')
    opts.addopts = '';
end
if ~isfield(opts, 'biasfield')
    opts.biasfield = false;
end

% set the output directories
corr_dir = sprintf('%s/N4-targets/', out_dir);
temp_dir = [out_dir, '/temp-out/'];

% first make sure the output directories exist
if ~exist(corr_dir, 'dir')
    mkdir(corr_dir);
end
if ~exist(temp_dir, 'dir')
    mkdir(temp_dir);
end


done_list = zeros([length(targets) 1]);
chk_files = cell([length(targets) 1]);
corr_targets = cell([length(targets) 1]);

% iterate over all of the targets
for j = 1:length(targets)


    % do some error checking
    if ~exist(targets{j}, 'file')
        error(sprintf('Target file %s does not exist', targets{j}));
    end

    % set the output filenames
    sfix = '.nii';
    [t_dir t_name t_ext] = fileparts(targets{j});
    if strcmp(t_name(end-3:end), '.nii')
        t_name = t_name(1:end-4);
        sfix = '.nii.gz';
    end

    corr_targets{j} = sprintf('%s%s%s', corr_dir, t_name, sfix);

    % let the user know what is going on
    tprintf('-> Running N4 Correction\n');
    tprintf('Target: %s\n', targets{j});
    tprintf('Corrected Target: %s\n', corr_targets{j});

    % make sure it needs to be run
    chk_files{j}{1} = corr_targets{j};
    if (opts.biasfield)
        corr_biasfield{j} = sprintf('%s%s_bias%s', corr_dir, t_name, sfix);
        chk_files{j}{2} = corr_biasfield{j};
    end
    if files_exist(chk_files{j})
        tprintf('Skipping N4 Correction: %s (output files exist)\n', t_name);
        tprintf('\n');
        pause(0.01);
        done_list(j) = 1;
        continue;
    end

    % create the output directory
    out_temp_dir = sprintf('%sN4-%s/', temp_dir, t_name);
    if ~exist(out_temp_dir, 'dir')
        mkdir(out_temp_dir);
    end


    % run the N4 correction
    if opts.biasfield
        cmds{1} = sprintf('%s -d 3 -i %s -o [%s,%s] %s \n', ...
                          N4loc, targets{j}, corr_targets{j}, ...
                          corr_biasfield{j}, opts.addopts);
    else
        cmds{1} = sprintf('%s -d 3 -i %s -o %s %s \n', ...
                          N4loc, targets{j}, corr_targets{j}, opts.addopts);
    end

    % run the commands
    txtout = [out_temp_dir, opts.txtout];
    if strcmp(runtype, 'cluster')
        pbsout = [out_temp_dir, opts.pbsfile];
        run_cmd_cluster(cmds, opts.memval, pbsout, txtout);
        done_list(j) = 0;
    else
        run_cmd_single(cmds, txtout);
        done_list(j) = 1;
    end

    tprintf('\n');

end
% iterate until done
while (min(done_list) == 0)
    for i = 1:length(done_list)
        if (done_list(i) == 0)
            if files_exist(chk_files{i})
                done_list(i) = 1;
            end
        end
    end
    pause(1);
end

