function [ Y ] = GetDataMatrix( folder, reduceTo, patchsize, totalImages )
%GETDATAMATRIX Summary of this function goes here
%   Detailed explanation goes here

fl = dir(folder);
Y = [];
% Auto calculate gap
% gap = 7;
for imindex = floor(linspace(3, 437, totalImages))
    imgTemp = imread([folder fl(imindex).name]);
    if size(imgTemp, 3) > 1
        imgTemp = rgb2gray(imgTemp);
    end
    imgTemp = imresize(imgTemp, reduceTo/min(size(imgTemp)));
    if size(imgTemp, 1) > reduceTo
        r = size(imgTemp, 1);
        drop = floor((r - reduceTo)/2);
        s = max(1, 1+drop - 1);
        imgTemp = imgTemp(s:(s+reduceTo-1),:);
    elseif size(imgTemp, 2) > reduceTo
        r = size(imgTemp, 2);
        drop = floor((r - reduceTo)/2);
        s = max(1, 1+drop - 1);
        imgTemp = imgTemp(:,s:(s+reduceTo-1));
    end
    Y = [Y im2patch(imgTemp, patchsize)];
    imshow(imgTemp)
    drawnow
    pause(0.1)
end
Y = Y./255;

end

