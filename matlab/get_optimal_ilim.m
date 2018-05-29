function ilim = get_optimal_ilim(img, seg, varargin)
% get_optimal_ilim - function to get optimal intensity limits for use in
%                    plot_segmentation_overlay
%
% Two forms:
% 1) ilim = get_optimal_ilim(img, seg)
% 2) ilim = get_optimal_ilim(img, seg, prctiles)
%
% Input: img - the intensity image
%        seg - the reference segmentation
%        prctiles - (optional) the percentiles to plot
%                 - default: [1, 99]
%
% Output: ilim - a two-element vector to be passed to plot_segmentation_overlay
%
%

if length(varargin) == 0
    prctiles = [1 99];
elseif length(varargin) == 1 && length(varargin{1}) == 2
    prctiles = varargin{1};
else
    error('Invalid input arguments');
end

mi_seg = prctile(img(seg > 0), prctiles(1));
mx_seg = prctile(img(seg > 0), prctiles(2));
mi_raw = min(img(:));
mx_raw = max(img(:));

ilim = zeros(2, 1);
ilim(1) = (mi_seg - mi_raw) / (mx_raw - mi_raw);
ilim(2) = (mx_seg - mi_raw) / (mx_raw - mi_raw);
