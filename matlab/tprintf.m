function tprintf(varargin)
% TPRINTF - same as fprintf except prepends time since last "tic"
%
% tprintf(varargin)
% 
% Input: typical printf format
% Output: none

try
    et = toc;
catch err
    fprintf(varargin{:});
    return;
end

hrs = floor(et / 3600);
et = rem(et, 3600);
mins = floor(et / 60);
secs = round(rem(et, 60));

fprintf('[%02dh %02dm %02ds] %s', hrs, mins, secs, sprintf(varargin{:}));
