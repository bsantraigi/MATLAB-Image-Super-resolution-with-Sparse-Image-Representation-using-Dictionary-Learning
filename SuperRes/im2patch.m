function [ p ] = im2patch( I, block)
%IM2PATCH Summary of this function goes here
%   Detailed explanation goes here

%% Overlapping
% w = block;
% h = block;
% p = zeros(block^2, (size(I, 1) - h + 1)*(size(I, 2) - w + 1));
% 
% pindex = 1;
% for i = 1:(size(I, 1) - h + 1)
%     for j = 1:(size(I, 2) - w + 1)
%         p(:, pindex) = reshape(I(i:(i + h - 1),j:(j+w-1)), w*h, 1);
%         pindex = pindex + 1;
%     end
% end

%% Non-Overlapping
w = block;
h = block;
p = zeros(block^2, (size(I, 1)/h)*(size(I, 2)/w));

pindex = 1;
for i = 1:size(I, 1)/h
    for j = 1:(size(I, 2)/w)
        x = (j - 1)*w + 1;
        y = (i - 1)*h + 1;
        p(:, pindex) = reshape(I(y:(y + h - 1),x:(x+w-1)), w*h, 1);
        p(:, pindex) = p(:, pindex) - mean(p(:, pindex));
        pindex = pindex + 1;
        
%         p(:, pindex) = reshape(I(y:(y + h - 1),x:(x+w-1)), w*h, 1);
%         pindex = pindex + 1;
    end
end

end

