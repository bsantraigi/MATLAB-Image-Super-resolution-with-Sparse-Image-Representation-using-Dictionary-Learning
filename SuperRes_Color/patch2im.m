function [ I, I2 ] = patch2im(...
    p, image_id, numrows, numcols, block,...
    Cb_of_Y, Cr_of_Y, overlap)
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
% w = block;
% h = block;
% 
% rows = floor(sqrt(size(p, 2)));
% I = zeros(rows*h, rows*w);
% pindex = 1;
% for i = 1:size(I, 1)/h
%     for j = 1:(size(I, 2)/w)
%         x = (j - 1)*w + 1;
%         y = (i - 1)*h + 1;
% %         p(:, pindex) = reshape(I(y:(y + h - 1),x:(x+w-1)), w*h, 1);
%         I(y:(y + h - 1),x:(x+w-1)) = reshape(p(:, pindex), h, w);
%         pindex = pindex + 1;
%     end
% end

%% Custom Overlap
% Supporting only square images
patchCols = block;
patchRows = block;
row_picks = 1:(patchRows - overlap):numrows;
col_picks = 1:(patchCols - overlap):numcols;
% Do zero padding to make sure no pixel on the edges gets missed
I = zeros(row_picks(end) + patchRows - 1,...
    col_picks(end) + patchCols - 1);
I2 = I;
% if row_picks(end) + patchRows - 1 > numrows
%     I((numrows + 1):(row_picks(end) + patchRows - 1), :) = 0;
% end
% 
% if col_picks(end) + patchCols - 1 > numcols
%     I(:, (numcols + 1):(col_picks(end) + patchCols - 1)) = 0;
% end

pindex = (image_id - 1)*length(row_picks)*length(col_picks) + 1;

for yrow = row_picks
    for xcol = col_picks
        pat = reshape(p(:, pindex), patchCols, patchRows);
        if xcol == 1 || yrow == 1
            I(yrow:(yrow + patchRows - 1),xcol:(xcol+patchCols-1)) = ...
                pat;
        else
            
            I((yrow + overlap - 1):(yrow + patchRows - 1),(xcol + overlap - 1):(xcol+patchCols-1)) = ...
                pat((overlap):end, (overlap):end);
        end
        I2(yrow:(yrow + patchRows - 1),xcol:(xcol+patchCols-1)) = ...
                pat;
        pindex = pindex + 1;
    end
end

I = I(1:numrows, 1:numcols);
I(:,:,1) = I;
I(:,:, 2) = Cb_of_Y(:,:, image_id);
I(:,:, 3) = Cr_of_Y(:,:, image_id);

I2 = I2(1:numrows, 1:numcols);
I2(:,:,1) = I2;
I2(:,:, 2) = Cb_of_Y(:,:, image_id);
I2(:,:, 3) = Cr_of_Y(:,:, image_id);
end

