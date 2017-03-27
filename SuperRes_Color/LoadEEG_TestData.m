function [ newData ] = LoadEEG_TestData( user )
%LOADEEG_DATA Summary of this function goes here
%   Detailed explanation goes here

f1 = sprintf('EEG_Kaggle/MatFiles_7Bands_&Meta/test_%d.mat', user);
f1 = load(f1);
newData = f1.newData;
for i = 0:(size(newData, 2)/29 - 1)
    newData(:, i*29 + 1) = mean(newData(:, (i*29 + 1):((i + 1)*29)), 2);
end
newData = newData(:, 1:29:size(newData, 2));

%% Normalize for all cases
newData = (newData - min(newData(:)))/(max(newData(:)) - min(newData(:))) - 0.5;

end

