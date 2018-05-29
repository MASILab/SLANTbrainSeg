function cams = render_3D_labels(raw, seg, varargin)
% RENDER_3D_LABELS - 3D rendering of a labeled segmentation image
%
% Form 1: render_3D_labels(raw, seg);
% Form 2: render_3D_labels(raw, seg, opts);
%
% Input: raw - the 3D raw (intensity) image
%        seg - the 3D segmentation (label) image
%        opts - (optional) the rendering options (struct)
%
% Output: None, however, a figure is created
%
% --- Available Options ---
%
% * General Options *
% opts.cr_buffer - The amount of buffering used when cropping (in voxels)
%                - The value -1 indicates no cropping
%                - (default: -1)
% opts.resdims - the resolution of the image (default: [1 1 1])
%
% * Slice Rendering Options *
% opts.ilim - the intensity limits for windowing (default: [0 1])
% opts.xslices - array of x-slices to render (default: [])
% opts.yslices - array of y-slices to render (default: [])
% opts.zslices - array of z-slices to render (default: [])
%              - Note: for opts.[xyz]slices if you set the value to 'mid'
%                      then the middle slice in that direction will
%                      be rendered
% opts.slicealpha - the transparency of the rendered slices (default: 0.6)
%
% * Label Rendering Options *
% opts.labels - array of label numbers to render (default: all unique labels)
% opts.labelsmooth - presmooth the labels with smooth3 (default: 0 = no)
% opts.isosmooth - smooth the isosurfaces (default {})
%                - see "smoothpatch" for details
% opts.labelcolors - the colors associated with the rendered labels
%                  - should be a L x 3 array where "L" is the number of labels
%                    specified by opts.labels
%                  - (default: hsv(L))
% opts.labelalphas - the transparency of the rendered labels
%                  - should be a L x 1 array with values between 0 and 1
%                  - (default: 0.8 for all labels)
% opts.bg_label - determines whether a background label should be rendered
%               - available values are 'none', 'all', and [tl th]
%               - if 'none', no background label will be rendered
%               - if 'all', background label will be seg > 0
%               - if [tl th], background label will be constructed by
%                 thresholding "raw" by raw > tl & raw < th
%               - (default: 'none')
% opts.bg_color - the color of the background label (default: [0 0 0])
% opts.bg_alpha - the transparency of the background label (default: 0.05)
%
% * Viewing Options *
% opts.fignum - the figure number of generate (default: 1, use -1 to use
%               the current figure without clearing)
% opts.lighting - the lighting type (default: 'phong') see "help lighting"
% opts.camlight - cell array of lighting (default: {'headlight'})
%               - see "help camlight"
% opts.material - the material type (default: 'dull')
% opts.azimuth - the azimuth angle (default: 90)
% opts.elevation - the elevation angle (default: 20)
%

% do some quick error checking
if length(varargin) > 1
    error('Invalid number of arguments to render3D_labels');
end
if ~isequal(size(raw), size(seg))
    error('Dimensions of "raw" and "seg" do not match');
end

% if they didn't specify any options then we will use all of the defaults
if isempty(varargin)
    opts = struct;
else
    opts = varargin{1};
end

% general options
if ~isfield(opts, 'cr_buffer'), opts.cr_buffer = -1; end
if ~isfield(opts, 'resdims'), opts.resdims = [1 1 1]; end

% slice rendering options
if ~isfield(opts, 'ilim'), opts.ilim = get_optimal_ilim(raw, seg); end
if ~isfield(opts, 'xslices'), opts.xslices = []; end
if ~isfield(opts, 'yslices'), opts.yslices = []; end
if ~isfield(opts, 'zslices'), opts.zslices = []; end
if ~isfield(opts, 'slicealpha'), opts.slicealpha = 0.6; end

% label rendering options
if ~isfield(opts, 'labels')
    opts.labels = unique(seg);
    opts.labels = opts.labels(2:end);
end
if ~isfield(opts, 'labelcolors')
    opts.labelcolors = hsv(length(opts.labels));
end

if ~isfield(opts, 'labelalphas')
    opts.labelalphas = 0.8 * ones(size(opts.labels));
end
if ~isfield(opts, 'labelsmooth')
    opts.labelsmooth = 0;
end
if ~isfield(opts, 'isosmooth'), opts.isosmooth = {}; end
if ~isfield(opts, 'bg_label'), opts.bg_label = 'none'; end
if ~isfield(opts, 'bg_color'), opts.bg_color = [0 0 0]; end
if ~isfield(opts, 'bg_alpha'), opts.bg_alpha = 0.05; end

% viewing options
if ~isfield(opts, 'fignum'), opts.fignum = 1; end
if ~isfield(opts, 'lighting'), opts.lighting = 'phong'; end
if ~isfield(opts, 'camlight'), opts.camlight = {'headlight'}; end
if ~isfield(opts, 'material'), opts.material = 'dull'; end
if ~isfield(opts, 'azimuth'), opts.azimuth = 90; end
if ~isfield(opts, 'elevation'), opts.elevation = 20; end

% make sure we're dealing with doubles
raw = double(raw);
seg = double(seg);

% set the background label
if strcmp(opts.bg_label, 'none')
    bgseg = double(zeros(size(seg)));
elseif strcmp(opts.bg_label, 'all')
    bgseg = double(seg > 0);
elseif length(opts.bg_label) == 2
    bgseg = double(raw > opts.bg_label(1) & raw < opts.bg_label(2));
else
    error('opts.bg_label should be "none", "all", or [threshlow threshhigh]');
end

% scale the intensity using opts.ilim
mi = min(raw(:));
mx = max(raw(:));
opts.ilim(opts.ilim < 0) = 0;
opts.ilim(opts.ilim > 1) = 1;
xl = mi + opts.ilim(1) * (mx - mi);
xh = mi + opts.ilim(2) * (mx - mi);
raw(raw < xl) = xl;
raw(raw > xh) = xh;
raw = (raw - xl) / (xh - xl);

% crop the image appropriately
if opts.cr_buffer == -1 % -1 means don't crop
    opts.cr_buffer = 1e6;
end
cr = determine_cropping_region(seg + bgseg, opts.cr_buffer);
tmpraw = raw(cr(1):cr(2), cr(3):cr(4), cr(5):cr(6));
tmpseg = seg(cr(1):cr(2), cr(3):cr(4), cr(5):cr(6));
tmpbgseg = bgseg(cr(1):cr(2), cr(3):cr(4), cr(5):cr(6));

% set the dimensions of rendered image/labels
dims = size(tmpraw)+2;

% set the cropped raw image
raw = ones(dims) * mean(tmpraw(:));
raw(2:(dims(1)-1), 2:(dims(2)-1), 2:(dims(3)-1)) = tmpraw;

% set the cropped label image
seg = zeros(dims);
seg(2:(dims(1)-1), 2:(dims(2)-1), 2:(dims(3)-1)) = tmpseg;

% set the cropped background label image
bgseg = zeros(dims);
bgseg(2:(dims(1)-1), 2:(dims(2)-1), 2:(dims(3)-1)) = tmpbgseg;

% clear away some variables we're done with
clear tmpraw;
clear tmpseg;

%
% adjust the slice numbers appropriately
%

tt = zeros(size(raw));
for l = 1:length(opts.labels)
    tt(seg == opts.labels(l)) = 1;
end
if isempty(find(tt > 0, 1))
    midX = round(dims(1) / 2);
    midY = round(dims(2) / 2);
    midZ = round(dims(3) / 2);
else
    [tX tY tZ] = ind2sub(size(raw), find(tt > 0));
    midX = round(mean(tX(:)));
    midY = round(mean(tY(:)));
    midZ = round(mean(tZ(:)));
end

% do x
if strcmp(opts.xslices, 'mid')
    xsl = midX;
else
    xsl = [];
    xr = cr(1):cr(2);
    for i = 1:length(opts.xslices)
        ind = find(opts.xslices(i) == xr);
        if ~isempty(ind)
            xsl = [xsl, ind(1)+1];
        end
    end
end
% do y
if strcmp(opts.yslices, 'mid')
    ysl = midY;
else
    ysl = [];
    yr = cr(3):cr(4);
    for i = 1:length(opts.yslices)
        ind = find(opts.yslices(i) == yr);
        if ~isempty(ind)
            ysl = [ysl, ind(1)+1];
        end
    end
end
% do z
if strcmp(opts.zslices, 'mid')
    zsl = midZ;
else
    zsl = [];
    zr = cr(5):cr(6);
    for i = 1:length(opts.zslices)
        ind = find(opts.zslices(i) == zr);
        if ~isempty(ind)
            zsl = [zsl, ind(1)+1];
        end
    end
end

% create the figure
if(opts.fignum>0), figure(opts.fignum); clf; end

% first, render the slices that we want
colormap(gray);
if sum(numel(ysl)+numel(xsl)+numel(zsl))>0
    ss = slice(1:dims(2), 1:dims(1), 1:dims(3), raw, ysl, xsl, zsl);
    for z = 1:length(ss)
        set(ss(z), 'FaceAlpha', opts.slicealpha);
    end
end

% second construct the background
if ~strcmp(opts.bg_label, 'none')
    
    if(opts.labelsmooth)
        iso = isosurface(smooth3(bgseg == 1), 0.5);
    else
        iso = isosurface(bgseg == 1, 0);
    end
    
    if ~isempty(opts.isosmooth)
        iso = smoothpatch(iso, opts.isosmooth{:});
    end
    
    num = size(iso.vertices, 1);
    patch('Vertices', iso.vertices, ...
        'Faces', iso.faces, ...
        'FaceVertexCData', repmat(opts.bg_color, [num 1]), ...
        'FaceColor', 'interp', ...
        'FaceAlpha', opts.bg_alpha, ...
        'EdgeColor', 'none');
end

% third, render the individual labels
for l = 1:length(opts.labels)
    
    % skip labels that will be invisible
    if (opts.labelalphas(l) == 0)
        continue;
    end
    
    if(opts.labelsmooth)
        iso = isosurface(smooth3(seg == opts.labels(l)), 0.5);
    else
        iso = isosurface(seg == opts.labels(l), 0);
    end
    
    if ~isempty(opts.isosmooth)
        iso = smoothpatch(iso, opts.isosmooth{:});
    end
    
    num = size(iso.vertices, 1);
    patch('Vertices', iso.vertices, ...
        'Faces', iso.faces, ...
        'FaceVertexCData', repmat(opts.labelcolors(l, :), [num 1]), ...
        'FaceColor', 'interp', ...
        'FaceAlpha', opts.labelalphas(l), ...
        'EdgeColor', 'none');
end

% set the viewing angle
view(opts.azimuth, opts.elevation);

% set the lighting options
lighting(opts.lighting);
cams = cell(size(opts.camlight));
for i = 1:length(opts.camlight)
    cams{i} = camlight(opts.camlight{i});
end
material(opts.material);

% set the final visualization options
shading interp;
daspect(1 ./ opts.resdims);
xlim([1 dims(2)]);
ylim([1 dims(1)]);
zlim([1 dims(3)]);
axis vis3d;
axis off;

if exist('ss','var')
    for z = 1:length(ss)
        set(ss(z), 'FaceLighting', 'none');
    end
end
