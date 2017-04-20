function [ Y, Cb_of_Y, Cr_of_Y ] = GetDataMatrix( ...
    folder, reduceTo, patchsize, imrange, overlap )
%GETDATAMATRIX Summary of this function goes here
%   Detailed explanation goes here

fl = dir(folder);
Y = [];
totalImages = length(imrange);
Cb_of_Y = zeros(reduceTo, reduceTo, totalImages);
Cr_of_Y = zeros(reduceTo, reduceTo, totalImages);
% Auto calculate gap
% gap = 7;
imx = 1;
% for imindex = floor(linspace(3, length(fl), totalImages))
for imindex = imrange + 2
    imgTemp = imread([folder fl(imindex).name]);
    fprintf('Reading %d: %s\n', imx, fl(imindex).name);
    % Resize
    imgTemp = imresize(imgTemp, ...
        reduceTo/min([size(imgTemp, 1) size(imgTemp, 2)]));
    if size(imgTemp, 1) > reduceTo
        r = size(imgTemp, 1);
        drop = floor((r - reduceTo)/2);
        s = max(1, 1+drop - 1);
        imgTemp = imgTemp(s:(s+reduceTo-1),:, :);
    elseif size(imgTemp, 2) > reduceTo
        r = size(imgTemp, 2);
        drop = floor((r - reduceTo)/2);
        s = max(1, 1+drop - 1);
        imgTemp = imgTemp(:, s:(s+reduceTo-1), :);
    end
    
    % Switch to YCbCr
    imgTemp = rgb2ycbcr(imgTemp);
    imgCb = imgTemp(:,:,2);
    imgCr = imgTemp(:,:,3);
    imgTemp = imgTemp(:,:,1);
    
    % Get data matrix and save other channels
    Y_temp = im2patch(imgTemp, patchsize, overlap);
    Y = [Y Y_temp];
    Cb_of_Y(:,:, imx) = imgCb;
    Cr_of_Y(:,:, imx) = imgCr;
    subplot(1,3,1)
    imshow(imgTemp)
    subplot(1,3,2)
    imshow(imgCb)
    subplot(1,3,3)
    imshow(imgCr)
    drawnow
    pause(0.1)
    imx = imx + 1;
end
Y = double(Y)/255;
Cb_of_Y = double(Cb_of_Y)/255;
Cr_of_Y = double(Cr_of_Y)/255;

end

