function [ p ] = im2patch( I, block)
%IM2PATCH Summary of this function goes here
%   Detailed explanation goes here
w = block;
h = block;
p = zeros(block^2, (size(I, 1) - h + 1)*(size(I, 2) - w + 1));

pindex = 1;
for i = 1:(size(I, 1) - h + 1)
    for j = 1:(size(I, 2) - w + 1)
        p(:, pindex) = reshape(I(i:(i + h - 1),j:(j+w-1)), w*h, 1);
        pindex = pindex + 1;
    end
end

end

