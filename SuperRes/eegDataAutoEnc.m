%% Actual Code
clear all
close all
user = 1;
sData = load(sprintf('EEG_Kaggle/user%d_Dictionaries_set1', user));
Y = LoadEEG_TestData(user);
% Y = LoadEEG_TrainData(1, 1, 2000, 'interictal');
imagesc(Y);
