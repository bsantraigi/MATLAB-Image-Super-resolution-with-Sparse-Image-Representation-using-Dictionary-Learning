function [ big_img ] = upSample_with_noise( small_img )
%UPSAMPLE_WITH_NOISE Summary of this function goes here
%   Detailed explanation goes here

Rows = size(small_img, 1);
Cols = size(small_img, 2);
big_img = zeros(Rows*2, Cols*2);
for i = 1:Rows
    big_img(i*2 - 1, :) = rand(1, 2*Cols);
%     big_img(i*2-1, 2:2:2*Cols) = small_img(i, 1:Cols);
%     big_img(i*2-1, 1:2:2*Cols) = small_img(i, 1:Cols);    
    big_img(i*2, 2:2:2*Cols) = rand(1, Cols);
%     big_img(i*2, 2:2:2*Cols) = small_img(i, 1:Cols);
    big_img(i*2, 1:2:2*Cols) = small_img(i, 1:Cols);    
end

Rows = size(small_img, 1);
Cols = size(small_img, 2);
big_img = zeros(Rows*2, Cols*2);
for i = 1:Rows
%     big_img(i*2 - 1, :) = rand(1, 2*Cols);
    big_img(i*2-1, 2:2:2*Cols) = small_img(i, 1:Cols);
    big_img(i*2-1, 1:2:2*Cols) = small_img(i, 1:Cols);    
%     big_img(i*2, 2:2:2*Cols) = rand(1, Cols);
    big_img(i*2, 2:2:2*Cols) = small_img(i, 1:Cols);
    big_img(i*2, 1:2:2*Cols) = small_img(i, 1:Cols);    
end

end

