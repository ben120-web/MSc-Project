function generatingNoisyEcgDatabase(noiseSignalPath, ...
    ecgSignalPath, ecgFs, AverageEcgLength, maxNosieSections, SNR, ...
    numberOfGeneratedNoisySignals)
% generatingNoisyEcgDatabase - estimate various noise signals at the
% desired SNR levels using the noise and ecg data stored in a given
% location. Reference noise signals are stored in
% ..\functions\datastore\referenceNoiseSignalGeneration folder or can be
% generated using referenceNoiseSignalGeneration.m. Result for each ecg
% data will be stored in a seperate file in a subfolder named
% simulatedNoiseData on the ecgsignalPath.
%
% Syntax: generatingNoisyEcgDatabase(noiseSignalPath, ...
%            ecgSignalPath, ecgFs, AverageEcgLength, maxNosieSections, SNR, ...
%            numberOfGeneratedNoisySignals)
%
% Inputs:
%    noiseSignalPath - Folder location of the reference noise signals.
%    ecgSignalPath - Folder location of the ecg data. Files with mat, edf
%       and ecg extension will be taken as files containing ecg data.
%
%    Optional:
%    ecgFs - Numeric scalar, sampling frequency in Hz. [Default: 500 Hz].
%    AverageEcgLength - Numeric scalar, ecg signal average length in
%       seconds. Only signals with +/- 3 seconds of this average length will
%       be processed. [Default: 30 s].
%    maxNosieSections - Numeric scalar, number of consecutive sections
%       taken from the reference noise signals. [Default: 2].
%    SNR -  A numeric vector, desired SNR levels in dB. [Default: [0, 6,
%       12, 18, 24]].
%    numberOfGeneratedNoisySignals - A numeric scalar, number of noisy
%       signals to be generated. [Default: 2].
%
% Output:
%    none.
%
% Example:
%    generatingNoisyEcgDatabase(noiseSignalPath, ecgSignalPath);
%    generatingNoisyEcgDatabase(noiseSignalPath, ecgSignalPath, 300, ...
%       62, 2, 0, 6);
%    generatingNoisyEcgDatabase(noiseSignalPath, ecgSignalPath, 300, ...
%       45, 5, [0, 6], 3);
%
% Other m-files required: validateStandardInput.m,
%                         bSecurOptimisedKnowledgePeakDetect.m,
%                         noiseSignalModeling.m.
% Subfunctions: computeRmsNoiseAmp,
%               generateInitialTable.
% MAT-files required: none

%------------- BEGIN CODE --------------
%% Set constants
DEFAULT_SNR = [0, 6, 12, 18, 24];
DEFAULT_Fs = 500;
QRS_SEARCH_WINDOW = 0.05; % [s]
DEFAULT_AVERAGE_LENGTH = 30; % [s].
DEFAULT_MAX_NOISE_SECTION = 2;
DEFAULT_NUMBER_OF_GENERATED_NOISY_SIGNALS = 2;
VALID_FILE_EXTENSIONS = {'mat', 'ecg', 'edf'};

%% Check inputs.
% Check correct number of arguments.
minArgs = 2;
maxArgs = 8;
narginchk(minArgs, maxArgs);

% Validate noiseSignalPath.
validateStandardInput(noiseSignalPath, 'scalarText', 'noiseSignalPath', 1);

if ~isfolder(noiseSignalPath)

    % Throw an error.
    ErrorStruct.message = ['Invalid noiseSignalPath the specified location ' ...
        'was not found.'];
    ErrorStruct.identifier = [mfilename,':invalidNoiseSignalPath'];
    error(ErrorStruct);

end

% Validate ecgSignalPath.
validateStandardInput(ecgSignalPath, 'scalarText', 'ecgSignalPath', 2);

if ~isfolder(ecgSignalPath)

    % Throw an error.
    ErrorStruct.message = ['Invalid ecgSignalPath the specified location ' ...
        'was not found.'];
    ErrorStruct.identifier = [mfilename,':invalidEcgSignalPath'];
    error(ErrorStruct);

end

% Validate ecgFs.
if nargin > 2 && ~isempty(ecgFs)

    validateStandardInput(ecgFs, 'fs', 'ecgFs', 3);

else

    ecgFs = DEFAULT_Fs;

end

% Validate AverageEcgLength.
if nargin > 3 && ~isempty(AverageEcgLength)

    validateattributes(AverageEcgLength, {'numeric'}, {'scalar', 'finite', ...
        'real', 'positive'}, mfilename, 'AverageEcgLength', 4);

else

    AverageEcgLength = DEFAULT_AVERAGE_LENGTH;

end

% Validate maxNosieSections.
if nargin > 4 && ~isempty(maxNosieSections)

    validateattributes(maxNosieSections, {'numeric'}, {'scalar', 'finite', ...
        'real', 'positive'}, mfilename, 'maxNosieSections', 5);

else

    maxNosieSections = DEFAULT_MAX_NOISE_SECTION;

end

% Validate SNR.
if nargin > 5 && ~isempty(SNR)

    validateattributes(SNR, {'numeric'}, {'vector', 'finite', ...
        'real'}, mfilename, 'maxNosieSections', 6);

else

    SNR = DEFAULT_SNR;

end

% Validate numberOfGeneratedNoisySignals.
if nargin > 6 && ~isempty(numberOfGeneratedNoisySignals)

    validateattributes(numberOfGeneratedNoisySignals, {'numeric'}, ...
        {'scalar', 'finite', 'real', 'positive'}, mfilename, ...
        'numberOfGeneratedNoisySignals', 7);

else

    numberOfGeneratedNoisySignals = DEFAULT_NUMBER_OF_GENERATED_NOISY_SIGNALS;

end

% Get the information from the ecg signal folder.
ecgSignalDirInfo = dir(fullfile(ecgSignalPath));

% Find the files with valid extension for ecg signal.
totoalNumberOfFiles = length(ecgSignalDirInfo);
validEcgFilesIndex = nan(totoalNumberOfFiles, 1);

for iFile = 3 : totoalNumberOfFiles

    currentFileExtension = ecgSignalDirInfo(iFile, 1).name(end-2:end);

    if ismember(currentFileExtension, VALID_FILE_EXTENSIONS)

        validEcgFilesIndex(iFile, 1) = iFile;

    end

end

% index of valid ecg signal files.
validEcgFilesIndex(isnan(validEcgFilesIndex)) = [];

% Get the information from the noise signals folder.
noiseSignalDirInfo = dir(fullfile(noiseSignalPath, '*mat'));

% Read stored noise signals from mat file.
initialNoiseData = load(fullfile(noiseSignalPath, noiseSignalDirInfo.name));

% Using the field names of the initial noise data find the valid noise
% signal. Field names that contains the word 'Noise' in them are considered
% valid noise signals.
noiseDataFieldNames = fieldnames(initialNoiseData);
nNoiseDataFieldNames = numel(noiseDataFieldNames);
validIndex = false(nNoiseDataFieldNames, 1);

for iNoiseDataFieldNames = 1 : numel(noiseDataFieldNames)

    validIndex(iNoiseDataFieldNames, 1) = ...
        contains(noiseDataFieldNames{iNoiseDataFieldNames}, "Noise");

end

validNoiseDataFieldNames = noiseDataFieldNames(validIndex);

% Arrange the noise signals in a matrix.
nNoises = numel(validNoiseDataFieldNames);
lengthOfNoiseSignal = max(size(initialNoiseData.(...
    validNoiseDataFieldNames{1, 1})));
noiseSignals = nan(lengthOfNoiseSignal, nNoises);

for iNoise = 1 : nNoises

    noiseSignals(:, iNoise) = initialNoiseData.(validNoiseDataFieldNames{iNoise, 1});

end

% Calculate the minimum and maximum length of ecg signals which will be
% used for noise generation.
acceptableEcgLength500 = [AverageEcgLength - 3, AverageEcgLength + 3];
acceptableEcgLength500 = acceptableEcgLength500.* DEFAULT_Fs;

% Calculate the maximum possible noise sections that are possible and
% update the maximum noise section value if required.
maxPossibleNoiseSections = floor(lengthOfNoiseSignal / acceptableEcgLength500(2));

if maxNosieSections > maxPossibleNoiseSections

    maxNosieSections = maxPossibleNoiseSections;

end

% Get the starting index of each noise section.
noiseSectionStart = 1 : acceptableEcgLength500(2) : lengthOfNoiseSignal;

% Generate all the noise signals for each section of data. These signals
% are not scaled with respect to SNR right now.
for iNoiseSection = 1 : maxNosieSections

    noiseSectionEnd = noiseSectionStart(iNoiseSection) + ...
        acceptableEcgLength500(2) - 1;

    % All noise signals for this section.
    thisNoiseSection = noiseSignals(noiseSectionStart(iNoiseSection) : ...
        noiseSectionEnd, :);

    isNoiseSignalPowerlineNoise = false;

    for jNoise = 1 : nNoises

        if contains(validNoiseDataFieldNames{jNoise, 1}, "power")

            isNoiseSignalPowerlineNoise = true;

        end

        initialGenerateNoise.(validNoiseDataFieldNames...
            {jNoise, 1}){iNoiseSection, 1} = noiseSignalModeling(...
            thisNoiseSection(:, jNoise), DEFAULT_Fs, ...
            numberOfGeneratedNoisySignals, isNoiseSignalPowerlineNoise);

    end

end

% Set the path of result folder to store the data.
resultDataFolder = fullfile(ecgSignalPath, 'simulatedNoiseData');

% Numer of valid ecg files.
nEcgFiles = numel(validEcgFilesIndex);

% If there are ecg files to process then make the result folder if
% required.
if nEcgFiles > 0 && ~isfolder(resultDataFolder)

    mkdir(resultDataFolder)

end

% Get the length of SNR vector.
nSNR = numel(SNR);

% Process each valid ecg file.
for iEcgFile = 1 : nEcgFiles

    ecgSignalFileName = fullfile(ecgSignalPath, ...
        ecgSignalDirInfo(validEcgFilesIndex(iEcgFile)).name);

    currentFileExtension = ...
        ecgSignalDirInfo(validEcgFilesIndex(iEcgFile)).name(end - 2 : end);

    % Read the ecg and fs information from the file based on the file
    % extension.
    if strcmp(currentFileExtension, 'mat')

        TempData = load(ecgSignalFileName);
        tempDataFieldNames = fieldnames(TempData);
        nTempDataFieldNames = numel(tempDataFieldNames);

        % If the file contains one variable it will be taken as the ecg
        % signal.
        if nTempDataFieldNames == 1

            rawEcgSignal = TempData.(tempDataFieldNames{1});

            % If the file has more than one variable we will check to see if it
            % has sampling frequency information. The biggest variable, in
            % length, will be taken as the ecg signal.
        elseif nTempDataFieldNames > 1

            isFsFieldExist = contains(lower(tempDataFieldNames), 'fs');

            if any(isFsFieldExist)

                updatedEcgFs = TempData.(tempDataFieldNames{isFsFieldExist});

            end

            tempDataLength = nan(nTempDataFieldNames, 1);

            for iFieldNames = 1 : nTempDataFieldNames

                tempDataLength(iFieldNames, 1) = ...
                    numel(TempData.(tempDataFieldNames{iFieldNames}));

            end

            [~, maxLengthIndex] = max(tempDataLength);
            rawEcgSignal = TempData.(tempDataFieldNames{maxLengthIndex});

        end

        % Read the data from '.edf' or '.ecg' files.
    else
        continue

    end

    % Validate the raw ecg signal and fs again just to catch any unusual
    % behaviour.
    validateStandardInput(rawEcgSignal, 'ecg', 'rawEcgSignal');
    validateStandardInput(updatedEcgFs, 'fs', 'updatedEcgFs');

    % Get the length of the raw ecg signal and calcualte the acceptable
    % length using updatedEcgFs.
    lengthOfRawEcgSignal = numel(rawEcgSignal);
    acceptableEcgLength = [AverageEcgLength - 3, AverageEcgLength + 3];
    acceptableEcgLength = acceptableEcgLength.* updatedEcgFs;

    % If the raw ecg signal is of acceptable length process the data to
    % scale the noise for each section with respect to SNR.
    if (lengthOfRawEcgSignal >= acceptableEcgLength(1)) && ...
            (lengthOfRawEcgSignal <= acceptableEcgLength(2))

        % Initial data table. One table will be generated for each valid
        % file and stored in a seperate file in result folder.
        [noiseNames, DataTable] = generateInitialTable(nSNR, SNR, maxNosieSections);

        % Resample the ecg signal to 500 Hz if required.
        if updatedEcgFs ~= DEFAULT_Fs

            ecgSignal = signalAwareResample(rawEcgSignal, DEFAULT_Fs, updatedEcgFs);

        else

            ecgSignal = rawEcgSignal;

        end

        % Make sure the ecg signal is in column orientation and store the
        % data and file name in the data table.
        ecgSignal = ecgSignal(:);
        DataTable.ecgSignal(1, 1) = {ecgSignal};
        DataTable.FileName(1, 1) = ...
            ecgSignalDirInfo(validEcgFilesIndex(iEcgFile)).name(1 : end - 4);

        % Get the length of the current ecg signal.
        lengthOfThisEcgSignal = numel(ecgSignal);

        % Detect the r-peak locations.
        [qrsLocations] = bSecurOptimisedKnowledgePeakDetect(ecgSignal, ...
            DEFAULT_Fs, ones(numel(ecgSignal), 1));

        % Convert the qrs search window to samples. This window will be used to
        % calculate the qrs amplitude.
        qrsSearchWindow = round(QRS_SEARCH_WINDOW * DEFAULT_Fs);

        % Per-allocate memory.
        nQrsLocations = numel(qrsLocations);
        qrsPeakToPeak = nan(nQrsLocations, 1);

        % Calculate the peak to peak amplitude of the qrs complexes.
        for jQrsLocations = 1 : nQrsLocations

            if (qrsLocations(jQrsLocations) + qrsSearchWindow <= ...
                    lengthOfThisEcgSignal) && (qrsLocations(jQrsLocations) - ...
                    qrsSearchWindow >= 1)

                qrsAmpMax = max(ecgSignal(qrsLocations(jQrsLocations) - ...
                    qrsSearchWindow : qrsLocations(jQrsLocations) + qrsSearchWindow));
                qrsAmpMin = min(ecgSignal(qrsLocations(jQrsLocations) - ...
                    qrsSearchWindow : qrsLocations(jQrsLocations) + qrsSearchWindow));
                qrsPeakToPeak(jQrsLocations, 1) = qrsAmpMax - qrsAmpMin;

            end

        end

        % Convert th peak to peak qrs amplitude to power. This is the correct
        % transformation for a sine wave. In this instance, it approximates close
        % enough for measurement. See the PDF of ecg SNR report for derivation.
        qrsPeakToPeakPower = (mean(qrsPeakToPeak, 'omitnan') ^ 2) / 8;

        % Now using the qrs peak to peak power all the noises in each
        % section will be scaled for every SNR level.
        for kSNR = 1 :nSNR

            for lNoise = 1 : nNoises

                thisNoise = initialGenerateNoise.(noiseDataFieldNames{lNoise, 1});

                for mSection = 1 : maxNosieSections

                    thisNoiseSection = thisNoise{mSection, 1};

                    for iGenSignal = 1 : numberOfGeneratedNoisySignals + 1

                        thisNoiseSignal = thisNoiseSection{iGenSignal, 1};

                        thisNoiseSignal = thisNoiseSignal(1 : lengthOfThisEcgSignal);

                        % Calculate power of noise signals.
                        thisNoiseSignalPower = ...
                            computeRmsNoiseAmp(thisNoiseSignal) ^ 2;

                        % Calculate the scale factor which is required to
                        % achieve the desired SNR level with each noisy
                        % signal.
                        scaleFactor = sqrt(qrsPeakToPeakPower / ...
                            (thisNoiseSignalPower * (10 ^ (SNR(kSNR) / 10))));

                        DataTable.(['SNR', num2str(SNR(kSNR))])(mSection, 1). ...
                            (noiseNames{lNoise, 1}){iGenSignal, 1} = ...
                            thisNoiseSignal .* scaleFactor;

                    end

                end

            end

        end

        % Generate the current data file name to store the results.
        dataFileName = fullfile(resultDataFolder, ...
            ecgSignalDirInfo(validEcgFilesIndex(iEcgFile)).name(1 : end - 4));

        % Save the results.
        save([dataFileName, '.mat'], 'DataTable', "DEFAULT_Fs");

    end

    updatedEcgFs = ecgFs;

end

end

%% Subfunction
function [noiseAmp] = computeRmsNoiseAmp(noiseSection)
% Computes the Root-Mean-Squared amplitude of a signal by measuring the RMS
% value for each second and then discarding the top and bottom 5% of
% values.

% In this function, sampling rate is fixed at 500 Hz.
SIGNAL_FS = 500; % [Hz]

% Get the length of the signal in complete seconds.
nSignalSeconds = floor(numel(noiseSection) / SIGNAL_FS);

% Calculate RMS value for each second.
rmsVals = nan(nSignalSeconds, 1);
for iSec = 1 : nSignalSeconds

    % Get indexes for start and end of section.
    thisSectionStart = (iSec - 1) * SIGNAL_FS + 1;
    thisSectionEnd = iSec * SIGNAL_FS;

    % Grab the section from the noise signal.
    thisSignalSection = noiseSection(thisSectionStart : thisSectionEnd);

    % Compute the RMS amplitude difference between the mean of the section
    % and the section itself.
    rmsVals(iSec) = rms(thisSignalSection - mean(thisSignalSection));

end

% Discard highest and lowest 5% of values.
noiseAmp = trimmean(rmsVals, 5, 'round');

end

%% Subfunction
function [noiseNames, DataTable] = generateInitialTable(nSNR, SNR, maxNosieSections)
% Create initial table.

variableNames = {'FileName'  'ecgSignal'  'SectionNum' 'SNR0'};
fileName = "";
ecgSignal = cell(1,1);
noiseStructure.bw = {};
noiseStructure.em = {};
noiseStructure.ma = {};
noiseStructure.pl50 = {};
noiseStructure.pl60 = {};


noiseNames = fieldnames(noiseStructure);
SectionNum = 1;
DataTable = table(fileName, ecgSignal, SectionNum, noiseStructure,...
    'VariableNames', variableNames);

nInitialVariableNames = numel(variableNames);

for i = 2 : nSNR

    variableNames{1, i + nInitialVariableNames - 1} = ['SNR', num2str(SNR(i))];

end

DataTable(1, nInitialVariableNames + 1 : nSNR + nInitialVariableNames - 1) = DataTable(1, 4);
DataTable.Properties.VariableNames = variableNames;
DataTable(maxNosieSections, :) = DataTable(1, :);
DataTable.SectionNum(1 : end, 1) = 1 : maxNosieSections;
DataTable.FileName(:, :) = "";

end
%------------- END OF CODE -------------