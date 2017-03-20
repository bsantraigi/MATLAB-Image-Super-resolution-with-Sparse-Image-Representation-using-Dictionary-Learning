function [ p, means_of_patches ] = im2patch( I, block, overlap)
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
% w = block;
% h = block;
% p = zeros(block^2, (size(I, 1)/h)*(size(I, 2)/w));
% 
% pindex = 1;
% for i = 1:size(I, 1)/h
%     for j = 1:(size(I, 2)/w)
%         x = (j - 1)*w + 1;
%         y = (i - 1)*h + 1;
%         p(:, pindex) = reshape(I(y:(y + h - 1),x:(x+w-1)), w*h, 1);
%         p(:, pindex) = p(:, pindex) - mean(p(:, pindex));
%         pindex = pindex + 1;
%         
% %         p(:, pindex) = reshape(I(y:(y + h - 1),x:(x+w-1)), w*h, 1);
% %         pindex = pindex + 1;
%     end
% end

%% Custom Overlap
numrows = size(I, 1);
numcols = size(I, 2);
patchCols = block;
patchRows = block;
row_picks = 1:(patchRows - overlap):numrows;
col_picks = 1:(patchCols - overlap):numcols;
% Do zero padding to make sure no pixel on the edges gets missed
if row_picks(end) + patchRows - 1 > numrows
    I((numrows + 1):(row_picks(end) + patchRows - 1), :) = 0;
end

if col_picks(end) + patchCols - 1 > numcols
    I(:, (numcols + 1):(col_picks(end) + patchCols - 1)) = 0;
end

p = zeros(block^2, length(row_picks)*length(col_picks));
means_of_patches = zeros(1, length(row_picks)*length(col_picks));
pindex = 1;

for yrow = row_picks
    for xcol = col_picks
        p(:, pindex) = reshape(I(yrow:(yrow + patchRows - 1),xcol:(xcol+patchCols-1)), patchCols*patchRows, 1);
        patch_mean = mean(p(:, pindex));
        means_of_patches(pindex) = patch_mean;
        p(:, pindex) = p(:, pindex) - patch_mean;
        pindex = pindex + 1;
    end
end

end

