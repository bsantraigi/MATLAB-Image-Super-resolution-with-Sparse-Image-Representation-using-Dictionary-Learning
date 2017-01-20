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
user = 3;
UserTable = TrainMeta(cellfun(@(x) x == user, TrainMeta.user), :);
fs = sprintf('D:/Bishal - MTP/neural_generative/EEG_Kaggle/user%d_Dictionaries_set1.mat', user);
fuser = load(fs);
train_data = fuser.S.*fuser.B;
train_labels = cell2mat(UserTable.preictal)';
train_labels = train_labels - ~train_labels;
train_data = [train_data; train_labels]';
%% Test data
ftest = load(sprintf('D:/Bishal - MTP/neural_generative/EEG_Kaggle/test_data_user%d.mat', user));
yfit = NN_user3(ftest.S.*ftest.B)';
yfit = 1./(1+exp(-yfit));
%% Import 0.54 AUC
filename = 'EEG_Kaggle/SVMSol.csv';
delimiter = ',';
startRow = 2;
formatSpec = '%s%f%[^\n\r]';
fileID = fopen(filename,'r');

dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines' ,startRow-1, 'ReturnOnError', false);

fclose(fileID);

rsol_File = dataArray{:, 1};
rsol_Class = dataArray{:, 2};
clearvars filename delimiter startRow formatSpec fileID dataArray ans;
filename = 'EEG_Kaggle/MatFiles_7Bands_&Meta/MetaFile_Test.csv';
delimiter = ',';
formatSpec = '%*s%s%f%*s%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'EmptyValue' ,NaN, 'ReturnOnError', false);
fclose(fileID);
fNames = dataArray{:, 1};
fIndex = dataArray{:, 2} + 28;
fIndex = fIndex./29;
% clearvars filename delimiter formatSpec fileID dataArray ans;
%% Import meta for user 2
% pData = cell(size(yfit, 1)/29,2);
resultMap = containers.Map('KeyType','char','ValueType','double');
j = 1;
for k = 1:length(rsol_File)
    resultMap(rsol_File{k}) = rsol_Class(k);
end
needle = sprintf('%d_', user)
for k = 1:length(fNames)
    fName = strtrim(fNames{k});
    if(strfind(fName, needle) > 0)
        i = fIndex(k);
%         fprintf('Replace "%s": %f with %f\n', fName, resultMap(fName), yfit(i));
        resultMap(fName) = yfit(i);
%         pData{j, 1} = fName;
% y        pData{j, 2} = yfit(i);
        j = j+1;
    end
end
solFile = fopen('SVMSol.csv', 'w');
fprintf(solFile, 'File,Class\n');
for i = 1:length(fNames)
    fName = strtrim(fNames{i});
    fprintf(solFile, '%s,%f\n', fName, resultMap(fName));
end
fclose(solFile);











