%% Auto-enc
clear all
close all
user = 3;
Y = LoadEEG_TrainData(user, 1, 4000, 'last');
% Y = sigmoid_Inv(Y);
imagesc(Y);
%%
autoenc = trainAutoencoder(Y, 60,...
    'MaxEpochs',400,...
    'EncoderTransferFunction', 'logsig',...
    'DecoderTransferFunction', 'logsig',...
    'SparsityRegularization',10,...
    'SparsityProportion',0.05);
Y_re = predict(autoenc, Y);
mserror = mse(Y_re)
%%
Y_test = LoadEEG_TestData(user);
Z_test = encode(autoenc, Y_test);
Y_test_re = decode(autoenc, Z);
% view(autoenc)
figure(2)
subplot(2,1,1)
imagesc(Y_test), colorbar;
subplot(2,1,2)
imagesc(Y_test_re), colorbar;
figure(3)
imagesc(Z)

%% Load Metadata
% Create a table TrainMeta
% Table has entries for filename and column-index

filename = 'D:/Bishal - MTP/neural_generative/EEG_Kaggle/MetaFile_Train.csv';
delimiter = ',';
formatSpec = '%*s%s%f%f%[^/n/r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'EmptyValue' ,NaN, 'ReturnOnError', false);
fclose(fileID);
TrainMeta = table(dataArray{1:end-1}, 'VariableNames', {'fname','start','finish'});
clearvars filename delimiter formatSpec fileID dataArray ans;

TrainMeta.index = TrainMeta.finish/29;
TrainMeta.user = cellfun(@(x) str2num(x(1)), TrainMeta.fname, 'UniformOutput', false);
% TrainMeta.preictal is the target value for any model
% Denotes whether a column corresponds to preictal(1) or interictal(0)
TrainMeta.preictal = cellfun(@(x) x((end-4):end), TrainMeta.fname, 'UniformOutput', false);
TrainMeta.preictal = cellfun(@(x) str2num(x(1)), TrainMeta.preictal, 'UniformOutput', false);
%% Prepare Data for user
UserTable = TrainMeta(cellfun(@(x) x == user, TrainMeta.user), :);
train_labels = cell2mat(UserTable.preictal)';
train_labels = train_labels - ~train_labels;
Z_train = encode(autoenc, Y);
Z_train = [Z_train', train_labels'];





