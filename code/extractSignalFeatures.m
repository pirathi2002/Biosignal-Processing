function output = extractSignalFeatures(signalTable, resultTable, windowSize, overlap, varargin)
%EXTRACTSIGNALFEATURES Comprehensive feature extraction with analysis and PCA
%   Now includes all practical requirements: band power, STFT, proper wavelets, 
%   Shannon entropy, z-score normalization, and visualization outputs
%
% New Parameters:
%   'fs' - Sampling frequency (required for band power)
%   'wavelet' - Wavelet type (default 'db4')
%   'bands' - Frequency bands for power calculation (cell array)

% Parse inputs
p = inputParser;
addRequired(p, 'signalTable', @istable);
addRequired(p, 'resultTable', @istable);
addRequired(p, 'windowSize', @(x) x > 0);
addRequired(p, 'overlap', @(x) x >= 0);
addParameter(p, 'correlThresh', 0.9, @(x) x > 0 && x < 1);
addParameter(p, 'pcaVariance', 0.95, @(x) x > 0 && x < 1);
addParameter(p, 'fs', 1, @(x) x > 0); % Default 1 Hz for normalized freq
addParameter(p, 'wavelet', 'db4', @ischar);
addParameter(p, 'bands', {[0.5,4], [4,8], [8,13], [13,30], [30,100]}, @iscell);
parse(p, signalTable, resultTable, windowSize, overlap, varargin{:});

% Validation
if height(signalTable) ~= height(resultTable)
    error('signalTable and resultTable must have same number of rows.');
end
if overlap >= windowSize
    error('Overlap must be less than window size.');
end

% Setup windowing
numRows = height(signalTable);
stepSize = windowSize - overlap;
startIndices = 1:stepSize:numRows;
numWindows = length(startIndices);

% Get numeric columns
isNumeric = varfun(@isnumeric, signalTable, 'OutputFormat', 'uniform');
numericCols = signalTable.Properties.VariableNames(isNumeric);

if isempty(numericCols)
    error('No numeric columns found in signalTable.');
end

% =================== PREALLOCATION FOR SPEED ===================
% --- 1. Pre-generate feature names and count total features ---
featureNames = {};
x_sample = zeros(windowSize, 1); % Dummy data to get name lists
fs = p.Results.fs;

for c = 1:length(numericCols)
    colName = numericCols{c};
    [~, timeNames] = extractTimeDomainFeatures(x_sample, colName);
    [~, freqNames] = extractFrequencyDomainFeatures(x_sample, colName, fs, p.Results.bands);
    [~, tfNames] = extractTimeFrequencyFeatures(x_sample, colName, fs, p.Results.wavelet);
    [~, advNames] = extractAdvancedFeatures(x_sample, colName);
    featureNames = [featureNames, timeNames, freqNames, tfNames, advNames];
end
numTotalFeatures = length(featureNames);

% --- 2. Preallocate the main feature matrix ---
allFeatures = zeros(numWindows, numTotalFeatures);

% --- 3. Create allResults table in one vectorized operation ---
allResults = resultTable(startIndices, :);
% =============================================================

% Process each window
for w = 1:numWindows
    startIdx = startIndices(w);
    endIdx = min(startIdx + windowSize - 1, numRows);
    windowData = signalTable(startIdx:endIdx, numericCols);
    
    featColIdx = 1; % Reset column index for each window/row
    
    % Extract features for each column
    for c = 1:length(numericCols)
        colName = numericCols{c};
        x = windowData.(colName);
        
        % Extract features
        [timeFeat, ~] = extractTimeDomainFeatures(x, colName);
        [freqFeat, ~] = extractFrequencyDomainFeatures(x, colName, fs, p.Results.bands);
        [tfFeat, ~] = extractTimeFrequencyFeatures(x, colName, fs, p.Results.wavelet);
        [advFeat, ~] = extractAdvancedFeatures(x, colName);
        
        % Combine and place into pre-allocated matrix
        combinedFeats = [timeFeat, freqFeat, tfFeat, advFeat];
        num_feats = length(combinedFeats);
        
        allFeatures(w, featColIdx : (featColIdx + num_feats - 1)) = combinedFeats;
        
        featColIdx = featColIdx + num_feats; % Update column index
    end
end

% Create feature table - PRIOR TO ANALYSIS
featTablePrior = array2table(allFeatures, 'VariableNames', featureNames);

% Feature correlation analysis
corrMatrix = corr(allFeatures, 'rows', 'complete');
[highCorrPairs, redundantFeats] = findHighCorrelations(corrMatrix, featureNames, p.Results.correlThresh);

% Feature selection - remove highly correlated features
keepFeats = ~ismember(1:length(featureNames), redundantFeats);
selectedFeatures = allFeatures(:, keepFeats);
selectedFeatureNames = featureNames(keepFeats);

% Z-score normalization (Step 6 requirement)
normalizedFeatures = zscore(selectedFeatures);
normalizedFeatures(isnan(normalizedFeatures)) = 0; % Handle constant features

% PCA analysis
[pcaCoeff, pcaScore, pcaLatent, ~, pcaExplained] = pca(normalizedFeatures, 'Rows', 'complete');
cumExplained = cumsum(pcaExplained);
optimalComponents = find(cumExplained >= p.Results.pcaVariance * 100, 1);

% Create feature table - AFTER PCA
pcaFeatureNames = cell(1, optimalComponents);
for i = 1:optimalComponents
    pcaFeatureNames{i} = sprintf('PC%d', i);
end
featTableAfterPCA = array2table(pcaScore(:, 1:optimalComponents), 'VariableNames', pcaFeatureNames);

% Create output structure with visualization data
output = struct();
output.featTablePrior = featTablePrior;
output.featTableAfterAnalysis = array2table(selectedFeatures, 'VariableNames', selectedFeatureNames);
output.featTableAfterPCA = featTableAfterPCA;
output.normalizedFeatures = array2table(normalizedFeatures, 'VariableNames', selectedFeatureNames);
output.resultTable = allResults;

% PCA structure with visualization data
output.pca = struct();
output.pca.coefficients = pcaCoeff;
output.pca.scores = pcaScore(:, 1:optimalComponents);
output.pca.scatterData = pcaScore(:, 1:3); % First 3 PCs for visualization
output.pca.eigenvalues = pcaLatent;
output.pca.explained = pcaExplained;
output.pca.cumExplained = cumExplained;
output.pca.optimalComponents = optimalComponents;
output.pca.featureNames = selectedFeatureNames;

% Analysis structure with visualization data
output.analysis = struct();
output.analysis.correlationMatrix = corrMatrix;
output.analysis.highCorrPairs = highCorrPairs;
output.analysis.removedFeatures = featureNames(redundantFeats);
output.analysis.selectedFeatures = selectedFeatureNames;
output.analysis.numOriginalFeatures = length(featureNames);
output.analysis.numSelectedFeatures = length(selectedFeatureNames);
output.analysis.numPCAComponents = optimalComponents;
output.analysis.varianceExplainedByPCA = cumExplained(optimalComponents);

% Generate visualization data
output.visualization = struct();
output.visualization.featureDistributions = generateDistributionData(allFeatures, featureNames);
output.visualization.psdData = generatePSDData(signalTable, numericCols, fs);

end

% ======================== HELPER FUNCTIONS ========================
% (All helper functions remain unchanged from the previous version)
% ... [Rest of the file is unchanged] ...

% ======================== MODIFIED FEATURE EXTRACTION FUNCTIONS ========================
function [features, names] = extractTimeFrequencyFeatures(x, colName, fs, waveletType)
% Time-frequency feature extraction with wavelets.
% The computationally expensive STFT (spectrogram) calculation has been removed.
features = [];
names = {};

% Wavelet Features (Step 4)
if length(x) >= 8
    level = 3;
    try
        [c, l] = wavedec(x, level, waveletType);
        
        % Wavelet energy at each level
        energy = zeros(1, level+1);
        for k = 1:level
            d = detcoef(c, l, k);
            energy(k) = sum(d.^2);
        end
        a = appcoef(c, l, waveletType, level);
        energy(level+1) = sum(a.^2);
        
        % Relative wavelet energy
        total_energy = sum(energy);
        rel_energy = energy / total_energy;
        
        % Wavelet entropy
        norm_energy = rel_energy(rel_energy > 0);
        wavelet_entropy = -sum(norm_energy .* log(norm_energy));
        
        features = [features, rel_energy, wavelet_entropy];
        names = [names, ...
            sprintf('wv_rel_energy1_%s', colName), ...
            sprintf('wv_rel_energy2_%s', colName), ...
            sprintf('wv_rel_energy3_%s', colName), ...
            sprintf('wv_rel_energy4_%s', colName), ...
            sprintf('wv_entropy_%s', colName)];
    catch
        features = [features, zeros(1, 5)];
        names = [names, ...
            sprintf('wv_rel_energy1_%s', colName), ...
            sprintf('wv_rel_energy2_%s', colName), ...
            sprintf('wv_rel_energy3_%s', colName), ...
            sprintf('wv_rel_energy4_%s', colName), ...
            sprintf('wv_entropy_%s', colName)];
    end
else
    features = [features, zeros(1, 5)];
    names = [names, ...
        sprintf('wv_rel_energy1_%s', colName), ...
        sprintf('wv_rel_energy2_%s', colName), ...
        sprintf('wv_rel_energy3_%s', colName), ...
        sprintf('wv_rel_energy4_%s', colName), ...
        sprintf('wv_entropy_%s', colName)];
end
end

function [features, names] = extractAdvancedFeatures(x, colName)
% Enhanced advanced features.
% The computationally expensive Sample Entropy calculation has been removed.
x = x(:);
N = length(x);

if N < 8
    features = zeros(1, 6); % Reduced from 7 to 6
    names = strcat({'wv1_', 'wv2_', 'wv3_', 'fractal_', 'nonlin_', 'shannon_'}, colName);
    return;
end

% Simplified wavelet features (kept for backward compatibility)
if N >= 2
    if mod(N, 2) == 1
        d1 = x(1:2:end-2) - x(2:2:end-1);
    else
        d1 = x(1:2:end-1) - x(2:2:end);
    end
    wv1 = std(d1);
else
    wv1 = 0;
end

if N >= 2
    if mod(N, 2) == 1
        x2 = (x(1:2:end-2) + x(2:2:end-1)) / 2;
    else
        x2 = (x(1:2:end-1) + x(2:2:end)) / 2;
    end
    wv2 = std(x2);
else
    wv2 = std(x);
end

if length(x2) >= 2
    if mod(length(x2), 2) == 1
        x3 = (x2(1:2:end-2) + x2(2:2:end-1)) / 2;
    else
        x3 = (x2(1:2:end-1) + x2(2:2:end)) / 2;
    end
    wv3 = std(x3);
else
    wv3 = wv2;
end

% Fractal dimension
fractalDim = calculateHiguchiFD(x);

% Nonlinearity measure
if mean(abs(x)) > 0
    nonlinearity = std(x) / mean(abs(x));
else
    nonlinearity = 0;
end

% Shannon entropy (Step 5)
[counts, ~] = histcounts(x, 'Normalization', 'probability');
counts = counts(counts > 0);
shannon = -sum(counts .* log2(counts));

% REMOVED: Sample entropy calculation was here.
features = [wv1, wv2, wv3, fractalDim, nonlinearity, shannon];
names = strcat({'wv1_', 'wv2_', 'wv3_', 'fractal_', 'nonlin_', 'shannon_'}, colName);
end

% ======================== UNCHANGED FUNCTIONS ========================
% The functions below this line have not been modified.

function [features, names] = extractFrequencyDomainFeatures(x, colName, fs, bands)
% Enhanced frequency domain features with band power
x = x(:);
N = length(x);
if N < 4
    features = zeros(1, 8 + length(bands)); 
    names = strcat({'medfreq_', 'meanfreq_', 'maxfreq_', 'specent_', 'specflat_', 'specroll_', 'speccentroid_', 'bandwidth_'}, colName);
    for b = 1:length(bands)
        names = [names, sprintf('bandpower_%d_%s', b, colName)];
    end
    return; 
end

% FFT and power spectral density
X = fft(x);
P = abs(X(1:floor(N/2)+1)).^2 / N;
f = (0:length(P)-1) * fs/N;

% Remove DC component
if length(P) > 1
    P_ac = P(2:end);
    f_ac = f(2:end);
else
    P_ac = P;
    f_ac = f;
end

totalPower = sum(P_ac);
if totalPower == 0 || isempty(P_ac)
    features = zeros(1, 8 + length(bands)); 
    names = strcat({'medfreq_', 'meanfreq_', 'maxfreq_', 'specent_', 'specflat_', 'specroll_', 'speccentroid_', 'bandwidth_'}, colName);
    for b = 1:length(bands)
        names = [names, sprintf('bandpower_%d_%s', b, colName)];
    end
    return; 
end

% Median frequency
cumPower = cumsum(P_ac);
halfPower = totalPower/2;
medianIdx = find(cumPower >= halfPower, 1);
if isempty(medianIdx)
    medianFreq = f_ac(end);
else
    medianFreq = f_ac(medianIdx);
end

% Mean frequency
meanFreq = sum(f_ac .* P_ac') / totalPower;

% Peak frequency
[~, maxIdx] = max(P_ac);
maxFreq = f_ac(maxIdx);

% Spectral entropy
Pnorm = P_ac / sum(P_ac);
Pnorm(Pnorm <= 0) = eps;
spectralEntropy = -sum(Pnorm .* log2(Pnorm)) / log2(length(Pnorm));

% Spectral flatness
spectralFlatness = exp(mean(log(P_ac + eps))) / mean(P_ac);

% Spectral rolloff (85%)
rolloffIdx = find(cumPower >= 0.85 * totalPower, 1);
if isempty(rolloffIdx), rolloffIdx = length(f_ac); end
spectralRolloff = f_ac(rolloffIdx);

% Spectral centroid
spectralCentroid = meanFreq;

% Spectral bandwidth
bandwidth = sqrt(sum(((f_ac - spectralCentroid).^2) .* P_ac') / totalPower);

% Band power calculation (Step 3)
bandPower = zeros(1, length(bands));
for b = 1:length(bands)
    band = bands{b};
    idx = f_ac >= band(1) & f_ac <= band(2);
    if any(idx)
        bandPower(b) = sum(P_ac(idx)) / totalPower;
    end
end

features = [medianFreq, meanFreq, maxFreq, spectralEntropy, spectralFlatness, ...
            spectralRolloff, spectralCentroid, bandwidth, bandPower];
names = strcat({'medfreq_', 'meanfreq_', 'maxfreq_', 'specent_', 'specflat_', ...
               'specroll_', 'speccentroid_', 'bandwidth_'}, colName);
for b = 1:length(bands)
    names = [names, sprintf('bandpower_%d_%s', b, colName)];
end
end

function distData = generateDistributionData(features, featureNames)
% Generate distribution data for histogram visualization
distData = struct();
numFeatures = size(features, 2);

for i = 1:min(20, numFeatures) % Limit to first 20 features
    feat = features(:, i);
    distData.(featureNames{i}) = struct(...
        'values', feat, ...
        'mean', mean(feat, 'omitnan'), ...
        'std', std(feat, 'omitnan'));
end
end

function psdData = generatePSDData(signalTable, numericCols, fs)
% Generate PSD data for visualization
psdData = struct();
for c = 1:min(5, length(numericCols)) % Limit to first 5 columns
    colName = numericCols{c};
    x = signalTable.(colName);
    
    if length(x) > 256
        [pxx, f] = pwelch(x, 256, 128, 256, fs);
        psdData.(colName) = struct('psd', pxx, 'freq', f);
    else
        psdData.(colName) = struct('psd', [], 'freq', []);
    end
end
end

function [features, names] = extractTimeDomainFeatures(x, colName)
% Time domain feature extraction
x = x(:);
N = length(x);

% Zero crossings (correct formula)
zc = sum(diff(sign(x)) ~= 0);

% Slope sign changes (correct implementation)
if N > 2
    ssc = sum(diff(sign(diff(x))) ~= 0);
else
    ssc = 0;
end

% Hjorth parameters (corrected formulas)
if N > 2
    dx = diff(x);
    ddx = diff(dx);
    activity = var(x);                    % Activity = variance of signal
    mobility = std(dx) / std(x);          % Mobility = std of 1st derivative / std of signal
    complexity = (std(ddx) / std(dx)) / mobility; % Complexity = (std of 2nd deriv / std of 1st deriv) / mobility
else
    activity = var(x);
    mobility = 0;
    complexity = 0;
end

features = [
    mean(x), std(x), rms(x), var(x), ...
    skewness(x), kurtosis(x), range(x), iqr(x), ...
    zc / (N-1), ...           % zero crossing rate (normalized)
    sum(abs(diff(x))), ...    % total variation
    mean(abs(diff(x))), ...   % mean absolute derivative
    activity, mobility, complexity
];

names = strcat({'mean_', 'std_', 'rms_', 'var_', 'skew_', 'kurt_', 'range_', ...
    'iqr_', 'zcr_', 'tv_', 'mad_', 'activity_', 'mobility_', 'complexity_'}, colName);
end

function sampen = calculateSampleEntropyOld(x, m, r)
% Sample entropy calculation (corrected algorithm)
N = length(x);
if N < m + 1
    sampen = 0; 
    return; 
end

% Template matching for patterns of length m and m+1
function phi = template_match(m)
    templates = zeros(N - m + 1, m);
    for i = 1:N - m + 1
        templates(i, :) = x(i:i + m - 1);
    end
    
    matches = 0;
    total_pairs = 0;
    
    for i = 1:size(templates, 1)
        for j = i + 1:size(templates, 1)  % Avoid self-matching
            if max(abs(templates(i, :) - templates(j, :))) <= r
                matches = matches + 1;
            end
            total_pairs = total_pairs + 1;
        end
    end
    
    if total_pairs > 0
        phi = matches / total_pairs;
    else
        phi = 0;
    end
end

% Calculate phi(m) and phi(m+1)
phi_m = template_match(m);
phi_m1 = template_match(m + 1);

% Sample entropy
if phi_m > 0 && phi_m1 > 0
    sampen = -log(phi_m1 / phi_m);
else
    sampen = 0;
end
end

function fd = calculateHiguchiFD(x)
% Higuchi's fractal dimension (simplified version)
x = x(:);
N = length(x);
kmax = min(8, floor(N/4));  % Maximum k value

if kmax < 2
    fd = 1.5;
    return;
end

k_values = 2:kmax;
L_k = zeros(size(k_values));

for idx = 1:length(k_values)
    k = k_values(idx);
    L_m = zeros(k, 1);
    
    for m = 1:k
        indices = m:k:N;
        if length(indices) < 2
            L_m(m) = 0;
            continue;
        end
        
        % Calculate curve length for this subsequence
        curve_length = sum(abs(diff(x(indices)))) * (N - 1) / ((length(indices) - 1) * k);
        L_m(m) = curve_length;
    end
    
    L_k(idx) = mean(L_m);
end

% Remove zero or invalid values
valid_idx = L_k > 0 & isfinite(L_k);
if sum(valid_idx) < 2
    fd = 1.5;
    return;
end

% Fit line in log-log space: log(L(k)) vs log(1/k)
log_k_inv = log(1 ./ k_values(valid_idx));
log_L = log(L_k(valid_idx));

% Linear regression
p = polyfit(log_k_inv, log_L, 1);
fd = p(1);  % Slope is the fractal dimension

% Ensure reasonable bounds
if fd < 1, fd = 1; end
if fd > 2, fd = 2; end
end

function [highCorrPairs, redundantFeatures] = findHighCorrelations(corrMatrix, featureNames, threshold)
% Find highly correlated feature pairs
[rows, cols] = find(abs(corrMatrix) > threshold & corrMatrix ~= 1);
highCorrPairs = {};
redundantFeatures = [];

for i = 1:length(rows)
    if rows(i) < cols(i) % Avoid duplicates
        pair = {featureNames{rows(i)}, featureNames{cols(i)}, corrMatrix(rows(i), cols(i))};
        highCorrPairs{end+1} = pair;
        redundantFeatures = [redundantFeatures, cols(i)];
    end
end

redundantFeatures = unique(redundantFeatures);
end

function sampen = calculateSampleEntropy(x, m, r)
%CALCULATESAMPLEENTROPY Calculates the Sample Entropy of a 1D time series.
%   sampen = calculateSampleEntropy(x, m, r) computes the Sample Entropy
%   of the input signal x.
%
%   Inputs:
%     x - Input time series (vector).
%     m - Embedding dimension (usually 2 or 3).
%     r - Tolerance level (typically 0.1 to 0.2 times the standard deviation of x).
%
%   Output:
%     sampen - Sample Entropy value. Returns 0 if calculation is not possible
%              (e.g., due to insufficient data or no matches).

x = x(:); % Ensure x is a column vector
N = length(x);

% Handle short signals or invalid parameters
if N < m + 2
    % Sample Entropy requires at least m+2 data points to form m+1 length patterns.
    % Specifically, for m=2, you need 4 points.
    sampen = 0; % Or NaN, depending on desired error handling
    return;
end

% Set r relative to standard deviation if it's not already
if r <= 0
    r = 0.2 * std(x); % Default tolerance if r is zero or negative
    if r == 0
        r = 0.1 * abs(mean(x)); % Fallback for constant signals
        if r == 0
            sampen = 0; % Cannot calculate for completely flat signal
            return;
        end
    end
end

% Helper function to calculate A and B (number of matches)
function [A, B] = count_matches(data, dim, tolerance)
    N_data = length(data);
    
    % Number of m-dimensional vectors
    if N_data < dim
        A = 0; B = 0; return;
    end
    
    count_B = 0; % Counter for matches of length dim
    count_A = 0; % Counter for matches of length dim + 1
    
    for i = 1:(N_data - dim) % Loop for starting points of template vectors
        template_i = data(i : i + dim - 1);
        
        for j = (i + 1) : (N_data - dim) % Loop for starting points of comparison vectors
            template_j = data(j : j + dim - 1);
            
            % Check if m-length patterns match
            if max(abs(template_i - template_j)) <= tolerance
                count_B = count_B + 1;
                
                % If m-length patterns match, check (m+1)-length patterns
                % Ensure we don't go out of bounds for (m+1)-length comparison
                if (i + dim) <= N_data && (j + dim) <= N_data
                    if abs(data(i + dim) - data(j + dim)) <= tolerance
                        count_A = count_A + 1;
                    end
                end
            end
        end
    end
    A = count_A;
    B = count_B;
end

% Calculate B (count of matches for length m)
[~, B] = count_matches(x, m, r);

% Calculate A (count of matches for length m+1)
[A, ~] = count_matches(x, m + 1, r);


% Avoid log(0)
if B == 0
    sampen = 0; % Or Inf, or NaN, depending on desired output for no matches
    warning('Sample entropy: No matches found for embedding dimension %d. Returning 0.', m);
    return;
end

if A == 0
    sampen = 0; % Or Inf, or NaN
    warning('Sample entropy: No matches found for embedding dimension %d+1. Returning 0.', m);
    return;
end

sampen = -log(A / B);

end