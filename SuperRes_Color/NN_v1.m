%% Read Training data meta file
filename = 'D:\Bishal - MTP\neural_generative\EEG_Kaggle\MatFiles_7Bands_&Meta\MetaFile_Train.csv';
delimiter = ',';
formatSpec = '%s%s%s%s%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);
fclose(fileID);
raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
for col=1:length(dataArray)-1
    raw(1:length(dataArray{col}),col) = dataArray{col};
end
numericData = NaN(size(dataArray{1},1),size(dataArray,2));

for col=[1,3,4]
    % Converts strings in the input cell array to numbers. Replaced non-numeric
    % strings with NaN.
    rawData = dataArray{col};
    for row=1:size(rawData, 1);
        % Create a regular expression to detect and remove non-numeric prefixes and
        % suffixes.
        regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
        try
            result = regexp(rawData{row}, regexstr, 'names');
            numbers = result.numbers;
            
            % Detected commas in non-thousand locations.
            invalidThousandsSeparator = false;
            if any(numbers==',');
                thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                if isempty(regexp(thousandsRegExp, ',', 'once'));
                    numbers = NaN;
                    invalidThousandsSeparator = true;
                end
            end
            % Convert numeric strings to numbers.
            if ~invalidThousandsSeparator;
                numbers = textscan(strrep(numbers, ',', ''), '%f');
                numericData(row, col) = numbers{1};
                raw{row, col} = numbers{1};
            end
        catch me
        end
    end
end
rawNumericColumns = raw(:, [1,3,4]);
rawCellColumns = raw(:, 2);
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),rawNumericColumns); % Find non-numeric cells
rawNumericColumns(R) = {NaN}; % Replace non-numeric cells
MetaFileTrain = raw;
clearvars filename delimiter formatSpec fileID dataArray ans raw col numericData rawData row regexstr result numbers invalidThousandsSeparator thousandsRegExp me rawNumericColumns rawCellColumns R;
%% 
% Data is loaded in the order [preictal, interictal]
pictalCols = [];
iictalCols = [];
user = 3;
needle1 = sprintf('%d_', user)
for i = 1:size(MetaFileTrain, 1)
    ps = strfind(MetaFileTrain{i, 2}, needle1);
    if sum(ps == 1) > 0
        if(~isempty(strfind(MetaFileTrain{i, 2}, '_1.mat')))
            pictalCols = [pictalCols MetaFileTrain{i, 4}/29];
%             pictalCols = [pictalCols; {MetaFileTrain{i, 2}}];
        end
        if(~isempty(strfind(MetaFileTrain{i, 2}, '_0.mat')))
            iictalCols = [iictalCols MetaFileTrain{i, 4}/29];
%             iictalCols = [iictalCols; {MetaFileTrain{i, 2}}];
        end
    end
end
pictalCols = pictalCols';
iictalCols = iictalCols';
nn_trainData = S.*B;















