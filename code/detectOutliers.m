function results = detectOutliers(dataTable, varargin)
% DETECTOUTLIERS Detect and handle outliers in table data
%
% results = detectOutliers(dataTable, varargin)
%
% This function detects outliers using statistical methods including Z-score,
% IQR method, and Modified Z-score with Median Absolute Deviation (MAD).
%
% INPUTS:
%   dataTable - MATLAB table containing numeric data
%
% OPTIONAL PARAMETERS (Name-Value pairs):
%   'Method' - Outlier detection method
%              'zscore' (default), 'iqr', 'modified_zscore'
%   'ZThreshold' - Z-score threshold for outlier detection (default: 3)
%   'IQRMultiplier' - IQR multiplier for outlier bounds (default: 1.5)
%   'MADThreshold' - Modified Z-score threshold (default: 3.5)
%   'ColumnsToProcess' - Cell array of column names to process (default: all numeric)
%   'Action' - What to do with outliers: 'flag' (default), 'winsorize', 'remove', 'replace_nan', 'median_replace'
%              'flag': Only identify outliers, don't modify data
%              'winsorize': Clip outliers to threshold boundaries (preserves signal continuity)
%              'median_replace': Replace outliers with local median (preserves signal continuity)
%              'replace_nan': Replace outliers with NaN (creates gaps)
%              'remove': Remove entire rows containing outliers
%   'WinsorizePercentile' - Percentile for winsorizing (default: 95)
%   'MedianWindowSize' - Window size for local median replacement (default: 5, must be odd)
%   'PreserveOriginal' - Keep original values in separate columns (default: false)
%
% OUTPUTS:
%   results - Structure containing:
%     .processedTable - Table with outliers handled according to 'Action'
%     .originalTable - Copy of the original input table
%     .outlierMask - Logical table showing outlier locations
%     .statistics - Detailed statistics for each column
%     .parameters - Processing parameters used
%     .summary - Overall summary statistics
%
% EXAMPLE:
%   % Create sample data with outliers
%   t = (1:100)';
%   signal = sin(2*pi*t/20) + 0.1*randn(100,1);
%   signal([10, 25, 50]) = [5, -4, 6]; % Add outliers
%   dataTable = table(t, signal, 'VariableNames', {'Time', 'Signal'});
%
%   % Detect and flag outliers using Modified Z-score (default action is 'flag')
%   results = detectOutliers(dataTable, 'Method', 'modified_zscore');
%
%   % Winsorize the outliers
%   results_winsorized = detectOutliers(dataTable, 'Method', 'modified_zscore', 'Action', 'winsorize');

% --- Main Function Body ---

% Parse input arguments
p = inputParser;
addRequired(p, 'dataTable', @(x) isa(x, 'table'));
addParameter(p, 'Method', 'zscore', @(x) ismember(x, {'zscore', 'iqr', 'modified_zscore'}));
addParameter(p, 'ZThreshold', 3, @(x) isnumeric(x) && x > 0);
addParameter(p, 'IQRMultiplier', 1.5, @(x) isnumeric(x) && x > 0);
addParameter(p, 'MADThreshold', 3.5, @(x) isnumeric(x) && x > 0);
addParameter(p, 'ColumnsToProcess', {}, @iscell);
addParameter(p, 'Action', 'flag', @(x) ismember(x, {'flag', 'remove', 'replace_nan', 'winsorize', 'median_replace'})); % Default changed to 'flag'
addParameter(p, 'WinsorizePercentile', 95, @(x) isnumeric(x) && x > 50 && x < 100);
addParameter(p, 'MedianWindowSize', 5, @(x) isnumeric(x) && x > 0 && mod(x,2) == 1);
addParameter(p, 'PreserveOriginal', false, @islogical);
parse(p, dataTable, varargin{:});

% Initialize results structure
results = struct();
results.originalTable = dataTable;
results.processedTable = dataTable;

% Determine columns to process
if isempty(p.Results.ColumnsToProcess)
    numericCols = {};
    for i = 1:width(dataTable)
        if isnumeric(dataTable{:,i})
            numericCols{end+1} = dataTable.Properties.VariableNames{i};
        end
    end
else
    numericCols = p.Results.ColumnsToProcess;
end

% Initialize outlier mask table (same structure as data table, but logical)
results.outlierMask = array2table(false(size(dataTable)), 'VariableNames', dataTable.Properties.VariableNames);
if ~isempty(dataTable.Properties.RowNames)
    results.outlierMask.Properties.RowNames = dataTable.Properties.RowNames;
end


% Initialize statistics structure
results.statistics = struct();
results.statistics.method = p.Results.Method;
results.statistics.columnsProcessed = numericCols;
results.statistics.outlierIndices = struct();
results.statistics.outlierCounts = struct();
results.statistics.detectionThresholds = struct();

% Process each numeric column
for i = 1:length(numericCols)
    colName = numericCols{i};
    originalData = results.processedTable{:, colName};
    
    % Store original data if requested
    if p.Results.PreserveOriginal
        results.processedTable.([colName '_Original']) = originalData;
    end
    
    % Detect outliers based on selected method
    switch lower(p.Results.Method)
        case 'zscore'
            [outlierIdx, threshold] = detectOutliersZScore(originalData, p.Results.ZThreshold);
        case 'iqr'
            [outlierIdx, threshold] = detectOutliersIQR(originalData, p.Results.IQRMultiplier);
        case 'modified_zscore'
            [outlierIdx, threshold] = detectOutliersModifiedZScore(originalData, p.Results.MADThreshold);
    end
    
    % Store outlier mask
    results.outlierMask{:, colName} = outlierIdx;
    
    % Apply action to outliers
    processedData = originalData;
    switch lower(p.Results.Action)
        case 'remove'
            % Will be handled at table level after processing all columns
        case 'replace_nan'
            processedData(outlierIdx) = NaN;
        case 'winsorize'
            processedData = winsorizeOutliers(processedData, outlierIdx, threshold);
        case 'median_replace'
            processedData = replaceWithLocalMedian(processedData, outlierIdx);
        case 'flag'
            % No modification to data, just flagged in mask
    end
    
    % Update processed table
    results.processedTable{:, colName} = processedData;
    
    % Store statistics
    results.statistics.outlierIndices.(colName) = find(outlierIdx);
    results.statistics.outlierCounts.(colName) = sum(outlierIdx);
    results.statistics.detectionThresholds.(colName) = threshold;
end

% Handle row removal if requested
if strcmp(p.Results.Action, 'remove')
    % Remove rows that have outliers in any processed column
    rowsWithOutliers = any(results.outlierMask{:, numericCols}, 2);
    
    results.processedTable(rowsWithOutliers, :) = [];
    results.statistics.removedRows = find(rowsWithOutliers);
    results.statistics.removedRowCount = sum(rowsWithOutliers);
end

% Store processing parameters
results.parameters = p.Results;

% Calculate summary statistics
results.summary = struct();
results.summary.totalColumns = width(dataTable);
results.summary.numericColumnsProcessed = length(numericCols);
results.summary.originalRows = height(dataTable);
results.summary.processedRows = height(results.processedTable);

% Total outliers across all columns
totalOutliers = sum(structfun(@(x) x, results.statistics.outlierCounts));
results.summary.totalOutliersDetected = totalOutliers;
if height(dataTable) > 0 && ~isempty(numericCols)
    results.summary.outlierPercentage = (totalOutliers / (height(dataTable) * length(numericCols))) * 100;
else
    results.summary.outlierPercentage = 0;
end


% --- Nested Helper Functions ---

% Helper function: Z-score outlier detection
function [outliers, threshold] = detectOutliersZScore(data, zThreshold)
    validData = data(~isnan(data));
    if numel(validData) < 2
        outliers = false(size(data));
        threshold = struct('method', 'zscore', 'threshold', zThreshold, 'mean', NaN, 'std', NaN);
        return;
    end
    
    mu = mean(validData);
    sigma = std(validData);
    
    threshold = struct('method', 'zscore', 'threshold', zThreshold, 'mean', mu, 'std', sigma);
    
    if sigma == 0
        outliers = false(size(data));
    else
        zScores = abs((data - mu) / sigma);
        outliers = zScores > zThreshold & ~isnan(data);
    end
end

% Helper function: IQR outlier detection
function [outliers, threshold] = detectOutliersIQR(data, multiplier)
    validData = data(~isnan(data));
    if numel(validData) < 4
        outliers = false(size(data));
        threshold = struct('method', 'iqr', 'multiplier', multiplier, 'Q1', NaN, 'Q3', NaN, 'IQR', NaN, 'lowerBound', NaN, 'upperBound', NaN);
        return;
    end
    
    Q1 = prctile(validData, 25);
    Q3 = prctile(validData, 75);
    IQR_val = Q3 - Q1;
    
    lowerBound = Q1 - multiplier * IQR_val;
    upperBound = Q3 + multiplier * IQR_val;
    
    threshold = struct('method', 'iqr', 'multiplier', multiplier, 'Q1', Q1, 'Q3', Q3, ...
                      'IQR', IQR_val, 'lowerBound', lowerBound, 'upperBound', upperBound);
    
    outliers = (data < lowerBound | data > upperBound) & ~isnan(data);
end

% Helper function: Modified Z-score outlier detection
function [outliers, threshold] = detectOutliersModifiedZScore(data, madThreshold)
    validData = data(~isnan(data));
    if numel(validData) < 2
        outliers = false(size(data));
        threshold = struct('method', 'modified_zscore', 'threshold', madThreshold, 'median', NaN, 'MAD', NaN);
        return;
    end
    
    medianVal = median(validData);
    MAD = median(abs(validData - medianVal));
    
    threshold = struct('method', 'modified_zscore', 'threshold', madThreshold, ...
                      'median', medianVal, 'MAD', MAD);
    
    if MAD == 0
        % If MAD is zero, fall back to a check for values not equal to the median
        outliers = (data ~= medianVal) & ~isnan(data);
    else
        modifiedZScores = 0.6745 * abs((data - medianVal) / MAD);
        outliers = modifiedZScores > madThreshold & ~isnan(data);
    end
end

% Helper function: Winsorize outliers (clip to threshold boundaries)
function processedData = winsorizeOutliers(data, outlierIdx, threshold)
    processedData = data;
    
    if ~any(outlierIdx)
        return;
    end
    
    % Determine clipping boundaries based on method used
    switch threshold.method
        case 'zscore'
            upperBound = threshold.mean + threshold.threshold * threshold.std;
            lowerBound = threshold.mean - threshold.threshold * threshold.std;
        case 'iqr'
            upperBound = threshold.upperBound;
            lowerBound = threshold.lowerBound;
        case 'modified_zscore'
            if threshold.MAD > 0
                factor = threshold.threshold / 0.6745;
                upperBound = threshold.median + factor * threshold.MAD;
                lowerBound = threshold.median - factor * threshold.MAD;
            else
                % If MAD is 0, use percentile approach as a fallback
                validData = data(~isnan(data));
                winsorizePercentile = p.Results.WinsorizePercentile;
                upperBound = prctile(validData, winsorizePercentile);
                lowerBound = prctile(validData, 100 - winsorizePercentile);
            end
    end
    
    % Clip outliers to boundaries
    processedData(data > upperBound & outlierIdx) = upperBound;
    processedData(data < lowerBound & outlierIdx) = lowerBound;
end

% Helper function: Replace outliers with local median
function processedData = replaceWithLocalMedian(data, outlierIdx)
    processedData = data;
    outlierIndices = find(outlierIdx);
    windowSize = p.Results.MedianWindowSize;
    halfWindow = floor(windowSize / 2);
    
    for k = 1:length(outlierIndices)
        idx = outlierIndices(k);
        
        % Define window boundaries
        startIdx = max(1, idx - halfWindow);
        endIdx = min(length(data), idx + halfWindow);
        
        % Extract window data excluding the outlier itself
        windowData = data(startIdx:endIdx);
        windowData(ismember(windowData, data(idx))) = []; % More robust removal of outlier value
        windowData = windowData(~isnan(windowData)); % Remove any NaN values
        
        % Replace with median if enough valid points exist
        if numel(windowData) >= 2 % Use local median if at least 2 other points are in the window
            processedData(idx) = median(windowData);
        else
            % Fallback: use median of the entire non-outlier dataset for that column
            allValidData = data(~outlierIdx & ~isnan(data));
            if ~isempty(allValidData)
                processedData(idx) = median(allValidData);
            end
        end
    end
end

end