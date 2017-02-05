clear all
user = 2;
dinter = load(sprintf('MatFiles_7Bands_&Meta/train_%d[interictal].mat', user));
dpre = load(sprintf('MatFiles_7Bands_&Meta/train_%d[preictal].mat', user));
li = size(dinter.newData, 2);
lp = size(dpre.newData, 2);
prob_preictal = lp/(lp + li);
li = min(li, 3*lp);
data = [dinter.newData(:, 1:li), dpre.newData(:, 1:lp)]';

labels = [zeros(li, 1); ones(lp, 1)];
dataCombo = [data labels];
% labels = [labels, ~labels];
%% FitCSVM
SVMModel = fitcsvm(data,labels,'KernelFunction','polynomial','Standardize',true, 'PolynomialOrder', 2);
% SVMModel = fitcsvm(data,labels,'KernelFunction','rbf','Standardize',true);
yCV = predict(SVMModel, data);
fprintf('Accuracy on train data %0.2f \n',sum(yCV == labels)/length(labels)*100);
%% FitCSVM - On test
ScoreSVMModel = fitPosterior(SVMModel,data,labels);
[yfit, yfitScore] = predict(ScoreSVMModel, dtest);
%% Quadratic SVM
dtest = load(sprintf('MatFiles_7Bands_&Meta/test_%d.mat', user));
dtest = [dtest.newData]';
[yfit, yscore] = quadSVM_u.predictFcn(dtest);

yfit = yfitScore(:,2);
for i = 1:size(yfit, 1)/29
    x = yfit(((i - 1)*29 + 1):(i*29));
%     yfit(((i - 1)*29 + 1):(i*29)) = sum(prob_preictal*x)/sum(prob_preictal*x + (1 - prob_preictal)*(~x));
    yfit(((i - 1)*29 + 1):(i*29)) = mean(x);
end
%% Import 0.54 AUC
filename = '/home/bishal/Documents/MTP/EEG_Kaggle/SVMSol.csv';
delimiter = ',';
startRow = 2;
formatSpec = '%s%f%[^\n\r]';
fileID = fopen(filename,'r');

dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines' ,startRow-1, 'ReturnOnError', false);

fclose(fileID);

rsol_File = dataArray{:, 1};
rsol_Class = dataArray{:, 2};
clearvars filename delimiter startRow formatSpec fileID dataArray ans;
filename = '/home/bishal/Documents/MTP/EEG_Kaggle/MatFiles_7Bands_&Meta/MetaFile_Test.csv';
delimiter = ',';
formatSpec = '%*s%s%f%*s%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'EmptyValue' ,NaN, 'ReturnOnError', false);
fclose(fileID);
fNames = dataArray{:, 1};
fIndex = dataArray{:, 2};
clearvars filename delimiter formatSpec fileID dataArray ans;
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