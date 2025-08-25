function reportData = generateFeatureExtractionReport(output, varargin)
%GENERATEFEATUREEXTRACTIONREPORT Generate comprehensive analysis report
%   Generates detailed report from extractSignalFeatures output including
%   feature analysis, PCA results, correlation analysis, and visualizations
%
%   INPUTS:
%       output - Output structure from extractSignalFeatures
%       varargin - Optional parameters:
%           'saveReport' - logical, save report to file (default: false)
%           'reportPath' - string, path for saving report (default: pwd)
%           'reportName' - string, report filename (default: 'FeatureReport')
%           'includeVisualizations' - logical, generate plots (default: true)
%
%   OUTPUTS:
%       reportData - Structure containing report sections and data

% Parse inputs
p = inputParser;
addRequired(p, 'output', @isstruct);
addParameter(p, 'saveReport', false, @islogical);
addParameter(p, 'reportPath', pwd, @ischar);
addParameter(p, 'reportName', 'FeatureReport', @ischar);
addParameter(p, 'includeVisualizations', true, @islogical);
parse(p, output, varargin{:});

% Initialize report structure
reportData = struct();
reportData.timestamp = datetime('now');
reportData.sections = {};

%% =================== EXECUTIVE SUMMARY ===================
fprintf('\n=== SIGNAL FEATURE EXTRACTION REPORT ===\n');
fprintf('Generated: %s\n\n', datestr(reportData.timestamp));

summary = struct();
summary.totalWindows = height(output.resultTable);
summary.originalFeatures = output.analysis.numOriginalFeatures;
summary.selectedFeatures = output.analysis.numSelectedFeatures;
summary.pcaComponents = output.analysis.numPCAComponents;
summary.varianceExplained = output.analysis.varianceExplainedByPCA;
summary.featuresRemoved = length(output.analysis.removedFeatures);

fprintf('EXECUTIVE SUMMARY:\n');
fprintf('- Total windows analyzed: %d\n', summary.totalWindows);
fprintf('- Original features extracted: %d\n', summary.originalFeatures);
fprintf('- Features after correlation filtering: %d (%.1f%% retained)\n', ...
    summary.selectedFeatures, 100*summary.selectedFeatures/summary.originalFeatures);
fprintf('- Features removed due to high correlation: %d\n', summary.featuresRemoved);
fprintf('- Optimal PCA components: %d\n', summary.pcaComponents);
fprintf('- Variance explained by PCA: %.2f%%\n\n', summary.varianceExplained);

reportData.summary = summary;

%% =================== FEATURE EXTRACTION ANALYSIS ===================
fprintf('FEATURE EXTRACTION ANALYSIS:\n');

% Analyze feature types
featureTypes = analyzeFeatureTypes(output.featTablePrior.Properties.VariableNames);
fprintf('Feature distribution by type:\n');
fprintf('- Time domain: %d features\n', featureTypes.timeDomain);
fprintf('- Frequency domain: %d features\n', featureTypes.frequencyDomain);
fprintf('- Time-frequency domain: %d features\n', featureTypes.timeFrequency);
fprintf('- Advanced features: %d features\n', featureTypes.advanced);

% Feature statistics
featStats = generateFeatureStatistics(output.featTablePrior);
fprintf('\nFeature quality assessment:\n');
fprintf('- Features with zero variance: %d\n', featStats.zeroVariance);
fprintf('- Features with missing values: %d\n', featStats.missingValues);
fprintf('- Mean feature range: %.2e\n', featStats.meanRange);
fprintf('- Features requiring normalization: %d\n\n', featStats.needNormalization);

reportData.featureAnalysis = struct();
reportData.featureAnalysis.types = featureTypes;
reportData.featureAnalysis.statistics = featStats;

%% =================== CORRELATION ANALYSIS ===================
fprintf('CORRELATION ANALYSIS:\n');
fprintf('High correlation pairs found: %d\n', length(output.analysis.highCorrPairs));

if ~isempty(output.analysis.highCorrPairs)
    fprintf('Top 10 highly correlated feature pairs:\n');
    numPairs = min(10, length(output.analysis.highCorrPairs));
    
    % Sort by correlation strength
    correlations = cellfun(@(x) abs(x{3}), output.analysis.highCorrPairs);
    [~, sortIdx] = sort(correlations, 'descend');
    
    for i = 1:numPairs
        pair = output.analysis.highCorrPairs{sortIdx(i)};
        fprintf('  %d. %s <-> %s (r=%.3f)\n', i, pair{1}, pair{2}, pair{3});
    end
    fprintf('\n');
end

% Correlation matrix statistics
corrStats = struct();
corrStats.meanCorrelation = mean(abs(output.analysis.correlationMatrix(:)));
corrStats.maxCorrelation = max(abs(output.analysis.correlationMatrix(output.analysis.correlationMatrix ~= 1)));
corrStats.highCorrPairs = length(output.analysis.highCorrPairs);

reportData.correlationAnalysis = corrStats;

%% =================== PCA ANALYSIS ===================
fprintf('PRINCIPAL COMPONENT ANALYSIS:\n');
fprintf('Components needed for %.0f%% variance: %d\n', ...
    output.pca.cumExplained(output.pca.optimalComponents), output.pca.optimalComponents);

fprintf('Variance explained by first 5 components:\n');
numShow = min(5, length(output.pca.explained));
for i = 1:numShow
    fprintf('  PC%d: %.2f%% (cumulative: %.2f%%)\n', ...
        i, output.pca.explained(i), output.pca.cumExplained(i));
end

% Analyze feature loadings
loadingAnalysis = analyzeFeatureLoadings(output.pca.coefficients, ...
    output.analysis.selectedFeatures, output.pca.optimalComponents);
fprintf('\nMost important features per component:\n');
for i = 1:min(3, output.pca.optimalComponents)
    fprintf('  PC%d top features: %s\n', i, ...
        strjoin(loadingAnalysis.topFeatures{i}(1:min(3, end)), ', '));
end
fprintf('\n');

reportData.pcaAnalysis = struct();
reportData.pcaAnalysis.explained = output.pca.explained;
reportData.pcaAnalysis.cumExplained = output.pca.cumExplained;
reportData.pcaAnalysis.loadings = loadingAnalysis;

%% =================== DATA QUALITY ASSESSMENT ===================
fprintf('DATA QUALITY ASSESSMENT:\n');

qualityMetrics = assessDataQuality(output);
fprintf('Overall data quality score: %.2f/10\n', qualityMetrics.overallScore);
fprintf('Quality breakdown:\n');
fprintf('  - Feature completeness: %.2f/10\n', qualityMetrics.completeness);
fprintf('  - Feature diversity: %.2f/10\n', qualityMetrics.diversity);
fprintf('  - Signal-to-noise ratio: %.2f/10\n', qualityMetrics.snr);
fprintf('  - Dimensionality efficiency: %.2f/10\n\n', qualityMetrics.efficiency);

reportData.qualityAssessment = qualityMetrics;

%% =================== RECOMMENDATIONS ===================
fprintf('RECOMMENDATIONS:\n');
recommendations = generateRecommendations(output, qualityMetrics);
for i = 1:length(recommendations)
    fprintf('%d. %s\n', i, recommendations{i});
end
fprintf('\n');

reportData.recommendations = recommendations;

%% =================== GENERATE VISUALIZATIONS ===================
if p.Results.includeVisualizations
    fprintf('Generating visualizations...\n');
    try
        figures = generateVisualizations(output);
        reportData.figures = figures;
        fprintf('Generated %d visualization figures\n\n', length(figures));
    catch ME
        fprintf('Warning: Could not generate all visualizations: %s\n\n', ME.message);
        reportData.figures = [];
    end
else
    reportData.figures = [];
end

%% =================== TECHNICAL DETAILS ===================
reportData.technicalDetails = struct();
reportData.technicalDetails.featureNames = output.analysis.selectedFeatures;
reportData.technicalDetails.removedFeatures = output.analysis.removedFeatures;
reportData.technicalDetails.pcaCoefficients = output.pca.coefficients;
reportData.technicalDetails.correlationMatrix = output.analysis.correlationMatrix;

%% =================== SAVE REPORT ===================
if p.Results.saveReport
    reportPath = fullfile(p.Results.reportPath, [p.Results.reportName '.mat']);
    save(reportPath, 'reportData');
    
    % Save text summary
    textPath = fullfile(p.Results.reportPath, [p.Results.reportName '_Summary.txt']);
    saveTextReport(reportData, textPath);
    
    fprintf('Report saved to: %s\n', p.Results.reportPath);
end

fprintf('Report generation completed.\n');
fprintf('==========================================\n\n');
end

%% =================== HELPER FUNCTIONS ===================

function featureTypes = analyzeFeatureTypes(featureNames)
% Analyze feature types based on naming convention
featureTypes = struct();
featureTypes.timeDomain = 0;
featureTypes.frequencyDomain = 0;
featureTypes.timeFrequency = 0;
featureTypes.advanced = 0;

timeKeywords = {'mean_', 'std_', 'rms_', 'var_', 'skew_', 'kurt_', 'range_', ...
                'iqr_', 'zcr_', 'tv_', 'mad_', 'activity_', 'mobility_', 'complexity_'};
freqKeywords = {'medfreq_', 'meanfreq_', 'maxfreq_', 'specent_', 'specflat_', ...
                'specroll_', 'speccentroid_', 'bandwidth_', 'bandpower_'};
tfKeywords = {'stft_', 'wv_rel_energy', 'wv_entropy'};
advKeywords = {'wv1_', 'wv2_', 'wv3_', 'sampen_', 'fractal_', 'nonlin_', 'shannon_'};

for i = 1:length(featureNames)
    name = featureNames{i};
    
    if any(cellfun(@(x) contains(name, x), timeKeywords))
        featureTypes.timeDomain = featureTypes.timeDomain + 1;
    elseif any(cellfun(@(x) contains(name, x), freqKeywords))
        featureTypes.frequencyDomain = featureTypes.frequencyDomain + 1;
    elseif any(cellfun(@(x) contains(name, x), tfKeywords))
        featureTypes.timeFrequency = featureTypes.timeFrequency + 1;
    elseif any(cellfun(@(x) contains(name, x), advKeywords))
        featureTypes.advanced = featureTypes.advanced + 1;
    end
end
end

function featStats = generateFeatureStatistics(featureTable)
% Generate comprehensive feature statistics
featData = table2array(featureTable);

featStats = struct();
featStats.zeroVariance = sum(var(featData, 'omitnan') == 0);
featStats.missingValues = sum(any(isnan(featData), 1));
featStats.meanRange = mean(range(featData, 'omitnan'));
featStats.needNormalization = sum(range(featData, 'omitnan') > 1000);

% Handle skewness and kurtosis with NaN values
skewVals = [];
kurtVals = [];
for i = 1:size(featData, 2)
    col = featData(:, i);
    col = col(~isnan(col)); % Remove NaN values
    if length(col) > 2
        skewVals(end+1) = skewness(col);
        kurtVals(end+1) = kurtosis(col);
    end
end

featStats.skewness = mean(abs(skewVals));
featStats.kurtosis = mean(kurtVals);
end

function loadingAnalysis = analyzeFeatureLoadings(coefficients, featureNames, numComponents)
% Analyze PCA loadings to identify most important features
loadingAnalysis = struct();
loadingAnalysis.topFeatures = cell(1, numComponents);

for i = 1:numComponents
    [~, sortIdx] = sort(abs(coefficients(:, i)), 'descend');
    topIdx = sortIdx(1:min(5, length(sortIdx)));
    loadingAnalysis.topFeatures{i} = featureNames(topIdx);
end
end

function qualityMetrics = assessDataQuality(output)
% Assess overall data quality
qualityMetrics = struct();

% Feature completeness (10 - percentage of features removed)
completeness = 10 * (1 - length(output.analysis.removedFeatures) / output.analysis.numOriginalFeatures);

% Feature diversity (based on feature type distribution)
typeCount = length(unique(cellfun(@(x) x(1:min(4,end)), output.analysis.selectedFeatures, 'UniformOutput', false)));
diversity = min(10, typeCount * 2.5);

% Signal-to-noise ratio approximation
featData = table2array(output.featTableAfterAnalysis);
% Handle NaN values properly
featMeans = mean(featData, 1, 'omitnan');
featStds = std(featData, 0, 1, 'omitnan');
validIdx = ~isnan(featMeans) & ~isnan(featStds) & featStds > 0;

if any(validIdx)
    snrApprox = mean(abs(featMeans(validIdx)) ./ featStds(validIdx));
    snr = min(10, max(0, 5 + log10(snrApprox + eps)));
else
    snr = 5; % Default middle score
end

% Dimensionality efficiency
efficiency = 10 * (output.analysis.numPCAComponents / output.analysis.numSelectedFeatures);

qualityMetrics.completeness = max(0, min(10, completeness));
qualityMetrics.diversity = max(0, min(10, diversity));
qualityMetrics.snr = max(0, min(10, snr));
qualityMetrics.efficiency = max(0, min(10, efficiency));
qualityMetrics.overallScore = mean([qualityMetrics.completeness, qualityMetrics.diversity, ...
                                   qualityMetrics.snr, qualityMetrics.efficiency]);
end

function recommendations = generateRecommendations(output, qualityMetrics)
% Generate actionable recommendations
recommendations = {};

% Based on feature removal
if length(output.analysis.removedFeatures) > 0.3 * output.analysis.numOriginalFeatures
    recommendations{end+1} = 'Consider reviewing signal preprocessing - high feature correlation suggests redundant information';
end

% Based on PCA results
if output.analysis.numPCAComponents < 0.1 * output.analysis.numSelectedFeatures
    recommendations{end+1} = 'Excellent dimensionality reduction achieved - consider using PCA features for modeling';
elseif output.analysis.numPCAComponents > 0.8 * output.analysis.numSelectedFeatures
    recommendations{end+1} = 'Limited dimensionality reduction - features may be too diverse or noisy';
end

% Based on data quality
if qualityMetrics.overallScore < 5
    recommendations{end+1} = 'Low data quality detected - consider signal filtering or feature selection refinement';
elseif qualityMetrics.overallScore > 8
    recommendations{end+1} = 'High data quality - features are well-suited for machine learning applications';
end

% Based on window count
if height(output.resultTable) < 100
    recommendations{end+1} = 'Small sample size - consider increasing data collection or reducing window overlap';
end

if isempty(recommendations)
    recommendations{1} = 'Feature extraction completed successfully with good quality metrics';
end
end

function figures = generateVisualizations(output)
% Generate key visualization figures
figures = struct();

try
    % PCA Variance Explained
    figure('Name', 'PCA Analysis', 'Position', [100, 100, 1200, 400]);
    
    subplot(1, 3, 1);
    bar(output.pca.explained(1:min(10, end)));
    title('PCA Variance Explained');
    xlabel('Principal Component');
    ylabel('Variance Explained (%)');
    grid on;
    
    subplot(1, 3, 2);
    plot(output.pca.cumExplained, 'o-', 'LineWidth', 2);
    hold on;
    yline(95, '--r', '95% Threshold');
    title('Cumulative Variance Explained');
    xlabel('Number of Components');
    ylabel('Cumulative Variance (%)');
    grid on;
    
    subplot(1, 3, 3);
    if size(output.pca.scatterData, 2) >= 3
        scatter3(output.pca.scatterData(:,1), output.pca.scatterData(:,2), ...
                 output.pca.scatterData(:,3), 36, 1:size(output.pca.scatterData,1), 'filled');
        title('PCA 3D Scatter Plot');
        xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
        colorbar;
        grid on;
    end
    
    figures.pcaFigure = gcf;
    
    % Correlation Matrix Heatmap
    if size(output.analysis.correlationMatrix, 1) <= 50  % Only for manageable size
        figure('Name', 'Feature Correlations', 'Position', [200, 200, 800, 600]);
        imagesc(abs(output.analysis.correlationMatrix));
        colorbar;
        title('Feature Correlation Matrix (Absolute Values)');
        xlabel('Feature Index');
        ylabel('Feature Index');
        
        figures.correlationFigure = gcf;
    end
    
catch ME
    warning(ME.identifier,'Some visualizations could not be generated: %s', ME.message);
end
end

function saveTextReport(reportData, filepath)
% Save a text summary of the report
fid = fopen(filepath, 'w');

fprintf(fid, 'SIGNAL FEATURE EXTRACTION REPORT\n');
fprintf(fid, 'Generated: %s\n\n', datestr(reportData.timestamp));

fprintf(fid, 'EXECUTIVE SUMMARY:\n');
fprintf(fid, '- Total windows: %d\n', reportData.summary.totalWindows);
fprintf(fid, '- Original features: %d\n', reportData.summary.originalFeatures);
fprintf(fid, '- Selected features: %d\n', reportData.summary.selectedFeatures);
fprintf(fid, '- PCA components: %d\n', reportData.summary.pcaComponents);
fprintf(fid, '- Variance explained: %.2f%%\n\n', reportData.summary.varianceExplained);

fprintf(fid, 'QUALITY SCORE: %.2f/10\n\n', reportData.qualityAssessment.overallScore);

fprintf(fid, 'RECOMMENDATIONS:\n');
for i = 1:length(reportData.recommendations)
    fprintf(fid, '%d. %s\n', i, reportData.recommendations{i});
end

fclose(fid);
end