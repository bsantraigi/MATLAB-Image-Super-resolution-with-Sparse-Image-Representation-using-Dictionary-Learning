function [ newData ] = LoadEEG_TrainData( user, start, till, type )
%LOADEEG_DATA Summary of this function goes here
%   Detailed explanation goes here

f1 = sprintf('EEG_Kaggle/MatFiles_7Bands_&Meta/train_%d[preictal].mat', user);
f2 = sprintf('EEG_Kaggle/MatFiles_7Bands_&Meta/train_%d[interictal].mat', user);

f1 = load(f1);
f2 = load(f2);
bands = 7;
if strcmp(type, 'both')
    newData = zeros(bands*16, till);
    r2 = randperm(size(f2.newData, 2), till/2);
    r1 = randperm(size(f1.newData, 2), till/2);

    newData(:, 1:(till/2)) = f2.newData(:, r2); % Interictal
    newData(:, (till/2 + 1):till) = f1.newData(:, r1); % Preictal
elseif strcmp(type, 'preictal')
    if till == 0
        newData = f1.newData(:, start:end);
    else
        newData = f1.newData(:, start:till);
    end
elseif strcmp(type, 'interictal')
    if till == 0
        newData = f2.newData(:, start:end);
    else
        newData = f2.newData(:, start:till);
    end
elseif strcmp(type, 'mean')
    newData = [f1.newData, f2.newData];
    for i = 0:(size(newData, 2)/29 - 1)
        newData(:, i*29 + 1) = mean(newData(:, (i*29 + 1):((i + 1)*29)), 2);
    end
    newData = newData(:, 1:29:size(newData, 2));
elseif strcmp(type, 'last')
    newData = [f1.newData, f2.newData];
    for i = 0:(size(newData, 2)/29 - 1)
        newData(:, i*29 + 1) = newData(:, (i + 1)*29);
    end
    newData = newData(:, 1:29:size(newData, 2));
end

%% Normalize for all cases
newData = (newData - min(newData(:)))/(max(newData(:)) - min(newData(:))) - 0.5;

end

