function results = imputeMissingValues(dataTable, varargin)
% IMPUTEMISSINGVALUES Handle missing values in table data
%
% results = imputeMissingValues(dataTable, varargin)
%
% This function handles missing values (NaN) in table data using various
% imputation methods including linear interpolation, spline interpolation,
% and shape-preserving PCHIP interpolation.
%
% INPUTS:
%   dataTable - MATLAB table containing numeric data with potential missing values
%
% OPTIONAL PARAMETERS (Name-Value pairs):
%   'Method' - Imputation method
%              'linear' (default), 'spline', 'pchip', 'delete', 'none'
%   'ColumnsToProcess' - Cell array of column names to process (default: all numeric)
%   'PreserveOriginal' - Keep original values in separate columns (default: false)
%   'ExtrapolationMethod' - How to handle boundary missing values
%                          'extrap' (default), 'nearest', 'none'
%
% OUTPUTS:
%   results - Structure containing:
%     .imputedTable - Table with missing values handled
%     .originalTable - Copy of the original input table
%     .missingMask - Logical table showing original missing value locations
%     .statistics - Detailed statistics for each column
%     .parameters - Processing parameters used
%     .summary - Overall summary statistics
%
% EXAMPLE:
%   % Create sample data with missing values
%   t = (1:100)';
%   signal = sin(2*pi*t/20) + 0.1*randn(100,1);
%   signal([15, 35, 75]) = NaN; % Add missing values
%   dataTable = table(t, signal, 'VariableNames', {'Time', 'Signal'});
%   
%   % Impute missing values using PCHIP
%   results = imputeMissingValues(dataTable, 'Method', 'pchip');

% Parse input arguments
p = inputParser;
addRequired(p, 'dataTable', @(x) isa(x, 'table'));
addParameter(p, 'Method', 'linear', @(x) ismember(x, {'linear', 'spline', 'pchip', 'delete', 'none'}));
addParameter(p, 'ColumnsToProcess', {}, @iscell);
addParameter(p, 'PreserveOriginal', false, @islogical);
addParameter(p, 'ExtrapolationMethod', 'extrap', @(x) ismember(x, {'extrap', 'nearest', 'none'}));

parse(p, dataTable, varargin{:});

% Initialize results structure
results = struct();
results.originalTable = dataTable;
results.imputedTable = dataTable;

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

% Initialize missing value mask table
results.missingMask = dataTable;
for i = 1:length(numericCols)
    results.missingMask{:, numericCols{i}} = false(height(dataTable), 1);
end

% Initialize statistics structure
results.statistics = struct();
results.statistics.method = p.Results.Method;
results.statistics.columnsProcessed = numericCols;
results.statistics.missingIndices = struct();
results.statistics.missingCounts = struct();
results.statistics.imputationSuccess = struct();

% Process each numeric column
for i = 1:length(numericCols)
    colName = numericCols{i};
    originalData = results.imputedTable{:, colName};
    
    % Store original data if requested
    if p.Results.PreserveOriginal
        results.imputedTable.([colName '_Original']) = originalData;
    end
    
    % Identify missing values
    missingIdx = isnan(originalData);
    results.missingMask{:, colName} = missingIdx;
    
    % Store missing value statistics
    results.statistics.missingIndices.(colName) = find(missingIdx);
    results.statistics.missingCounts.(colName) = sum(missingIdx);
    
    % Apply imputation method
    if any(missingIdx) && ~strcmp(p.Results.Method, 'none')
        switch lower(p.Results.Method)
            case 'delete'
                % Will be handled at table level after processing all columns
                imputedData = originalData;
                success = true;
                
            case 'linear'
                [imputedData, success] = imputeLinear(originalData, p.Results.ExtrapolationMethod);
                
            case 'spline'
                [imputedData, success] = imputeSpline(originalData, p.Results.ExtrapolationMethod);
                
            case 'pchip'
                [imputedData, success] = imputePCHIP(originalData, p.Results.ExtrapolationMethod);
        end
    else
        imputedData = originalData;
        success = ~any(missingIdx); % Success if no missing values to begin with
    end
    
    % Update imputed table
    results.imputedTable{:, colName} = imputedData;
    results.statistics.imputationSuccess.(colName) = success;
end

% Handle deletion method at table level
if strcmp(p.Results.Method, 'delete')
    % Remove rows with any NaN values in processed columns
    rowsWithNaN = false(height(results.imputedTable), 1);
    for i = 1:length(numericCols)
        rowsWithNaN = rowsWithNaN | isnan(results.imputedTable{:, numericCols{i}});
    end
    
    results.imputedTable(rowsWithNaN, :) = [];
    results.statistics.deletedRows = find(rowsWithNaN);
    results.statistics.deletedRowCount = sum(rowsWithNaN);
end

% Store processing parameters
results.parameters = struct();
results.parameters.method = p.Results.Method;
results.parameters.columnsToProcess = p.Results.ColumnsToProcess;
results.parameters.preserveOriginal = p.Results.PreserveOriginal;
results.parameters.extrapolationMethod = p.Results.ExtrapolationMethod;

% Calculate summary statistics
results.summary = struct();
results.summary.totalColumns = width(dataTable);
results.summary.numericColumnsProcessed = length(numericCols);
results.summary.originalRows = height(dataTable);
results.summary.imputedRows = height(results.imputedTable);

% Total missing values across all columns
totalMissing = 0;
successfulImputations = 0;
for i = 1:length(numericCols)
    colName = numericCols{i};
    totalMissing = totalMissing + results.statistics.missingCounts.(colName);
    if results.statistics.imputationSuccess.(colName)
        successfulImputations = successfulImputations + 1;
    end
end

results.summary.totalMissingValues = totalMissing;
results.summary.missingPercentage = (totalMissing / (height(dataTable) * length(numericCols))) * 100;
results.summary.successfulImputations = successfulImputations;
results.summary.imputationSuccessRate = (successfulImputations / length(numericCols)) * 100;

end

% Helper function: Linear interpolation imputation
function [imputedData, success] = imputeLinear(data, extrapolationMethod)
    imputedData = data;
    missingIdx = isnan(data);
    success = false;
    
    if sum(~missingIdx) < 2
        return; % Cannot interpolate with less than 2 valid points
    end
    
    validIdx = find(~missingIdx);
    validData = data(validIdx);
    
    try
        % Choose extrapolation method
        if strcmp(extrapolationMethod, 'none')
            extrapMethod = NaN; % No extrapolation
        else
            extrapMethod = extrapolationMethod;
        end
        
        % Interpolate missing values
        imputedData(missingIdx) = interp1(validIdx, validData, find(missingIdx), 'linear', extrapMethod);
        success = true;
    catch
        success = false;
    end
end

% Helper function: Spline interpolation imputation
function [imputedData, success] = imputeSpline(data, extrapolationMethod)
    imputedData = data;
    missingIdx = isnan(data);
    success = false;
    
    if sum(~missingIdx) < 4
        % Fall back to linear if insufficient points for spline
        [imputedData, success] = imputeLinear(data, extrapolationMethod);
        return;
    end
    
    validIdx = find(~missingIdx);
    validData = data(validIdx);
    
    try
        % Choose extrapolation method
        if strcmp(extrapolationMethod, 'none')
            extrapMethod = NaN; % No extrapolation
        else
            extrapMethod = extrapolationMethod;
        end
        
        % Interpolate missing values using spline
        imputedData(missingIdx) = interp1(validIdx, validData, find(missingIdx), 'spline', extrapMethod);
        success = true;
    catch
        success = false;
    end
end

% Helper function: PCHIP interpolation imputation
function [imputedData, success] = imputePCHIP(data, extrapolationMethod)
    imputedData = data;
    missingIdx = isnan(data);
    success = false;
    
    if sum(~missingIdx) < 2
        return; % Cannot interpolate with less than 2 valid points
    end
    
    validIdx = find(~missingIdx);
    validData = data(validIdx);
    
    try
        % Choose extrapolation method
        if strcmp(extrapolationMethod, 'none')
            extrapMethod = NaN; % No extrapolation
        else
            extrapMethod = extrapolationMethod;
        end
        
        % Interpolate missing values using PCHIP
        imputedData(missingIdx) = interp1(validIdx, validData, find(missingIdx), 'pchip', extrapMethod);
        success = true;
    catch
        success = false;
    end
end