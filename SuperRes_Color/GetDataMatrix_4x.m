function [ Y, means_of_Y ] = GetDataMatrix( ...
    folder, reduceTo, patchsize, totalImages, overlap )
%GETDATAMATRIX Summary of this function goes here
%   Detailed explanation goes here

fl = dir(folder);
Y = [];
means_of_Y = [];
% Auto calculate gap
% gap = 7;
for imindex = floor(linspace(3, length(fl), totalImages))
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
    imgTemp = imresize(imgTemp, 0.5, 'bicubic');
    imgTemp = imresize(imgTemp, 2, 'bicubic');
    [Y_temp, means_of_Y_temp] = im2patch(imgTemp, patchsize, overlap);
    Y = [Y Y_temp];
    means_of_Y = [means_of_Y means_of_Y_temp];
    imshow(imgTemp)
    drawnow
    pause(0.1)
end
Y = Y./255;
means_of_Y = means_of_Y./255;

end

