function generatingNoisyEcgDatabase(noiseSignalPath, ...
    ecgSignalPath, ecgFs, AverageEcgLength, maxNosieSections, SNR, ...
    numberOfGeneratedNoisySignals)

%------------- BEGIN CODE --------------
%% Set constants
DEFAULT_SNR = [0, 6, 12, 18, 24];
DEFAULT_Fs = 500;
QRS_SEARCH_WINDOW = 0.05; % [s]
DEFAULT_AVERAGE_LENGTH = 30; % [s].
DEFAULT_MAX_NOISE_SECTION = 2;
DEFAULT_NUMBER_OF_GENERATED_NOISY_SIGNALS = 2;

% Get the information from the noise signals folder.
noiseSignalDirInfo = dir(fullfile(noiseSignalPath, '*mat'));
ecgSignalDirInfo = dir(fullfile(ecgSignalPath, '*mat'));

% Read stored noise signals from mat file.
initialNoiseData = load(fullfile(noiseSignalPath, noiseSignalDirInfo.name));

% Calculate the minimum and maximum length of ecg signals which will be
% used for noise generation.
acceptableEcgLength500 = [AverageEcgLength - 3, AverageEcgLength + 3];
acceptableEcgLength500 = acceptableEcgLength500.* DEFAULT_Fs;

lengthOfNoiseSignal = floor(numel(initialNoiseData.emNoise) ./ initialNoiseData.SIGNAL_FS);

noiseSignal = initialNoiseData.emNoise;

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
    thisNoiseSection = noiseSignal(noiseSectionStart(iNoiseSection) : ...
        noiseSectionEnd, :);

    initialGenerateNoise.("EM_Noise"){iNoiseSection, 1} = noiseSignalModeling(...
        thisNoiseSection, DEFAULT_Fs, ...
        numberOfGeneratedNoisySignals);


end

% Set the path of result folder to store the data.
resultDataFolder = fullfile(erase(ecgSignalPath, 'cleanSignals'), 'simulatedNoiseData');

if ~isfolder(resultDataFolder) mkdir(resultDataFolder); end % Make directory.

% Numer of valid ecg files.
nEcgFiles = height(ecgSignalDirInfo);

% Get the length of SNR vector.
nSNR = numel(SNR);

% Process each valid ecg file.
for iEcgFile = 1 : nEcgFiles

    ecgSignalFileName = fullfile(ecgSignalPath, ...
        ecgSignalDirInfo(iEcgFile).name);

    TempData = load(ecgSignalFileName);
    tempDataFieldNames = fieldnames(TempData);

    % If the file contains one variable it will be taken as the ecg
    % signal.
    rawEcgSignal = TempData.(tempDataFieldNames{1});

    % Get the length of the raw ecg signal and calcualte the acceptable
    % length using updatedEcgFs.
    lengthOfRawEcgSignal = numel(rawEcgSignal);
    acceptableEcgLength = [AverageEcgLength - 3, AverageEcgLength + 3];
    acceptableEcgLength = acceptableEcgLength .* DEFAULT_Fs;

    % If the raw ecg signal is of acceptable length process the data to
    % scale the noise for each section with respect to SNR.
    if (lengthOfRawEcgSignal >= acceptableEcgLength(1)) && ...
            (lengthOfRawEcgSignal <= acceptableEcgLength(2))

        % Initial data table. One table will be generated for each valid
        % file and stored in a seperate file in result folder.
        [noiseNames, DataTable] = generateInitialTable(nSNR, SNR, maxNosieSections);

        % Make sure the ecg signal is in column orientation and store the
        % data and file name in the data table.
        ecgSignal = ecgSignal(:);
        DataTable.ecgSignal(1, 1) = {ecgSignal};
        DataTable.FileName(1, 1) = ...
            ecgSignalDirInfo(iEcgFile).name(1 : end - 4);

        % Get the length of the current ecg signal.
        lengthOfThisEcgSignal = numel(ecgSignal);

        % Detect the r-peak locations.
        qrsLocations = TempData.(tempDataFieldNames{2});

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

        % Convert the peak to peak qrs amplitude to power. This is the correct
        % transformation for a sine wave. In this instance, it approximates close
        % enough for measurement. See the PDF of ecg SNR report for derivation.
        qrsPeakToPeakPower = (mean(qrsPeakToPeak, 'omitnan') ^ 2) / 8;

        % Now using the qrs peak to peak power all the noises in each
        % section will be scaled for every SNR level.
        for kSNR = 1 : nSNR

            thisNoise = initialGenerateNoise.(noiseDataFieldNames);

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
                        (noiseNames){iGenSignal, 1} = ...
                        thisNoiseSignal .* scaleFactor;

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