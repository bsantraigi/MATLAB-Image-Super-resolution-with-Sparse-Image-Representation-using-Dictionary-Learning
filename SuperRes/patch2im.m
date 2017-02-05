function [ I ] = patch2im( p, block)
%PATCH2IMAGE Summary of this function goes here
%   Only supports square shaped image reconstruction
%   e.g. 64x64
w = block;
h = block;
rows = floor(sqrt(size(p, 2))) + h -1;
I = zeros(rows, rows);
pindex = 1;
for i = 1:(rows - h + 1)
    for j = 1:(rows - w + 1)
        I(i:(i + h - 1),j:(j+w-1)) = reshape(p(pindex:(pindex + w*h - 1)), w, h);
        pindex = pindex + w*h;
    end
end

end

