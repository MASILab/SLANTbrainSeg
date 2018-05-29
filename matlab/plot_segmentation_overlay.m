function varargout = plot_segmentation_overlay(im, seg, alpha, cmap, intlim, sl)
% PLOT_SEGMENTATION_OVERLAY - plots a segmented image overlayed with the
%                             raw image (uses the current figure)
%
% plot_segmentation_overlay(im, seg)
% plot_segmentation_overlay(im, seg, alpha)
% plot_segmentation_overlay(im, seg, alpha, cmap)
% plot_segmentation_overlay(im, seg, alpha, cmap, intlim)
% plot_segmentation_overlay(im, seg, alpha, cmap, intlim, sl)
%  
% Note: Any of the optional arguments can be left empty for default., 
%      eg.: plot_segmentation_overlay(im, seg, [], [], [], sl)
%
% Input: im - the raw image (a raw slice)
%        seg - the segmented image (again, a slice)
%        alpha - number between 0 and 1 [optional, default = 0.3]
%                0 => all raw image
%                1 => all segmented image
%        cmap - the colormap to use [optional,default = jet]
%        intlim - intensity limits for grayscale transformation
%                 [optional,default [min, max] of im]
%        sl - the slice to visualize [optional,default = middle slice]
%
% Output: [optional] rgb image generated and used in function
%

if nargin<6 || isempty(sl);
    sl = round(size(im,3)/2);
end
if nargin<5 || isempty(intlim);
    intlim = [0, 1];
end
if nargin<4 || isempty(cmap);
    cmap = jet;SegNii
end
if nargin<3 || isempty(alpha);
    alpha = .3;
end
if nargin < 2;
    error('need to supply 1st and 2nd (intenity and segmentation) arguments');
end
if ~isequal(size(im), size(seg));
    error('intensity and segmentation are different sizes.');
end

% remap seg to be consecutive label numbers
seg = uint16(seg);
max_label_num = max(seg(:))+1;
ul = unique(seg);
ll = zeros(max_label_num, 1);
ll(ul+1) = 0:1:(length(ul)-1);
seg(:) = ll(seg(:)+1);
max_label_num = max(seg(:))+1;

im = double(im);
mi = min(im(:));
mx = max(im(:));
im = im(:, :, sl);

seg = uint16(seg(:, :, sl));

% give option to hardcode intensity limits
if length(intlim)==3 && intlim(1) == -1;
    % make the rescaling constants not rescale
    mi=0;
    mx=1;
     % change size of intlim back to what rest of funct expects them to be
    intlim = [intlim(2) intlim(3)];
end

% fix the intensity limits
intlim(intlim < 0) = 0;
intlim(intlim > 1) = 1;

% convert the raw image to an rgb image
xl = mi + intlim(1) * (mx - mi);
xh = mi + intlim(2) * (mx - mi);
im(im < xl) = xl;
im(im > xh) = xh;
im = (im - xl) / (xh - xl);
rgb = repmat(im, [1 1 3]);

% resample the colormap
cs = size(cmap, 1);
inds = round(linspace(1, cs, max_label_num));
cmap = cmap(inds, :);

% convert the segmented image to an rgb image
clr = double(ind2rgb(seg, cmap));

% create the overlay image
overlayim = rgb;
inds = find(seg > 0);
for ch = 1:3
    rgbch = rgb(:, :, ch);
    overlaych = overlayim(:, :, ch);
    clrch = clr(:, :, ch);
    overlaych(inds) = alpha * clrch(inds) + (1 - alpha) * rgbch(inds);
    overlayim(:, :, ch) = overlaych;
end

% resized_img = imresize(overlayim, [2000, 2000]);
resized_img = overlayim;

% plot it
if nargout == 0;
    imagesc(resized_img);
    axis equal;
    axis off;
end

% optional output arg
if nargout>0
    varargout{1} = resized_img;
end
