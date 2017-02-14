function [ I ] = patch2im( p, block)
%PATCH2IMAGE Summary of this function goes here
%   Only supports square shaped image reconstruction
%   e.g. 64x64

%% Overlapping
% w = block;
% h = block;
% rows = floor(sqrt(size(p, 2))) + h -1;
% I = zeros(rows, rows);
% pindex = 1;
% for i = 1:(rows - h + 1)
%     for j = 1:(rows - w + 1)
%         I(i:(i + h - 1),j:(j+w-1)) = reshape(p(pindex:(pindex + w*h - 1)), w, h);
%         pindex = pindex + w*h;
%     end
% end

%% Non-Overlapping
w = block;
h = block;

rows = floor(sqrt(size(p, 2)));
I = zeros(rows*h, rows*w);
pindex = 1;
for i = 1:size(I, 1)/h
    for j = 1:(size(I, 2)/w)
        x = (j - 1)*w + 1;
        y = (i - 1)*h + 1;
%         p(:, pindex) = reshape(I(y:(y + h - 1),x:(x+w-1)), w*h, 1);
        I(y:(y + h - 1),x:(x+w-1)) = reshape(p(:, pindex), h, w);
        pindex = pindex + 1;
    end
end

end

