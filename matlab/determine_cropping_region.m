function cr = determine_cropping_region(truth, buffer)
% DETERMINE_CROPPING_REGION - takes in a volume and a buffer and returns an
%                             array that indicates the cropping region for
%                             the provided volume
%
% cr = determine_cropping_region(truth, buffer)
%
% Input: truth - the volume to analyze
%        buffer - the amount of space to leave in all directions
%
% Output: cr - the cropping region of the form [sx ex sy ey sz ey], where
%                   - sx -> starting x index
%                   - ex -> ending x index
%                   - sy -> starting y index
%                   - ex -> ending y index
%                   - sz -> starting z index
%                   - ez -> ending z index
%

% if given empty mask
if ~any(truth);
    cr = nan(1,6);
    return
end

% the cropping region
cr = zeros(1, 6);

% iteraate in all directions to find the boundaries
i = 1;
cont = 1;
while cont
    if max(max(truth(i, :, :))) > 0
        cr(1) = max([i - buffer, 1]);
        cont = 0;
    end
    i = i + 1;
end

i = size(truth, 1);
cont = 1;
while cont
    if max(max(truth(i, :, :))) > 0
        cr(2) = min([i + buffer, size(truth, 1)]);
        cont = 0;
    end
    i = i - 1;
end

i = 1;
cont = 1;
while cont
    if max(max(truth(:, i, :))) > 0
        cr(3) = max([i - buffer, 1]);
        cont = 0;
    end
    i = i + 1;
end

i = size(truth, 2);
cont = 1;
while cont
    if max(max(truth(:, i, :))) > 0
        cr(4) = min([i + buffer, size(truth, 2)]);
        cont = 0;
    end
    i = i - 1;
end

i = 1;
cont = 1;
while cont
    if max(max(truth(:, :, i))) > 0
        cr(5) = max([i - buffer, 1]);
        cont = 0;
    end
    i = i + 1;
end

i = size(truth, 3);
cont = 1;
while cont
    if max(max(truth(:, :, i))) > 0
        cr(6) = min([i + buffer, size(truth, 3)]);
        cont = 0;
    end
    i = i - 1;
end

