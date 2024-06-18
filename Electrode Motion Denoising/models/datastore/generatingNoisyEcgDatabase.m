function generatingNoisyEcgDatabase(noiseSignalPath, ...
    ecgSignalPath, ecgFs, maxNosieSections, SNR, ...
    numberOfGeneratedNoisySignals)
% generatingNoisyEcgDatabase - Models new noise sections, then corrupts the
% clean ECG's with pre-defined SNR levels of electrode motion noise.
%
% Syntax: generatingNoisyEcgDatabase(noiseSignalPath, ...
%    ecgSignalPath, ecgFs, maxNosieSections, SNR, ...
%    numberOfGeneratedNoisySignals)
%
% Inputs:
%    noiseSignalPath - Directory to real noise .mat file.
%    ecgSignalPath - Directory to all clean ECG signals.
%    ecgFs - Sampling frequency of ECG signals.
%    maxNosieSections - Maximum number of noise segments per ECG file.
%    SNR - Array containing the required signal to noise ratios.
%    numberOfGeneratedNoisySignals - The number of noisy signals to
%    generate from each clean ECG signal.
%
% Outputs: none.
%
% Other m-files required: noiseSignalModeling.m.
% Subfunctions: computeRmsNoiseAmp, generateInitialTable.
% MAT-files required: none.
%
%------------- BEGIN CODE --------------
%% Set constants
QRS_SEARCH_WINDOW = 0.05; % [s]
ECG_LENGTH_SECONDS = 30;

% Get the information from the noise signals folder.
noiseSignalDirInfo = dir(fullfile(noiseSignalPath, '*mat'));
ecgSignalDirInfo = dir(fullfile(ecgSignalPath, '*mat'));

% Read stored noise signals from mat file.
initialNoiseData = load(fullfile(noiseSignalPath, noiseSignalDirInfo.name));

% Calculate the minimum and maximum length of ecg signals which will be
% used for noise generation.
acceptableEcgLength500 = ECG_LENGTH_SECONDS;
acceptableEcgLength500 = acceptableEcgLength500 .* ecgFs;

lengthOfNoiseSignal = numel(initialNoiseData.emNoise);

% Define the noise signal.
noiseSignal = initialNoiseData.emNoise;

% Get the starting index of each noise section.
noiseSectionStart = 1 : acceptableEcgLength500 : lengthOfNoiseSignal;

% Generate all the noise signals for each section of data. These signals
% are not scaled with respect to SNR right now.
for iNoiseSection = 1 : maxNosieSections

    noiseSectionEnd = noiseSectionStart(iNoiseSection) + ...
        acceptableEcgLength500 - 1;

    % All noise signals for this section.
    thisNoiseSection = noiseSignal(noiseSectionStart(iNoiseSection) : ...
        noiseSectionEnd, :);

    initialGenerateNoise.("EM_Noise"){iNoiseSection, 1} = noiseSignalModeling(...
        thisNoiseSection, ...
        numberOfGeneratedNoisySignals);


end

% Set the path of result folder to store the data.
resultDataFolder = fullfile(erase(ecgSignalPath, 'cleanSignals'), 'trainingDataSet');

if ~isfolder(resultDataFolder) mkdir(resultDataFolder); end % Make directory.

% Filter of meta data 
filterFlag = string({ecgSignalDirInfo.name}) == "AngleData.mat" | ...
    string({ecgSignalDirInfo.name}) == "widthData.mat" | ...
    string({ecgSignalDirInfo.name}) == "zData.mat";


ecgSignalDirInfo(filterFlag) = [];

% Numer of valid ecg files.
nEcgFiles = height(ecgSignalDirInfo);

% Get the length of SNR vector.
nSNR = numel(SNR);

% Process each valid ecg file.
for iEcgFile = 1 : nEcgFiles - 3

    ecgSignalFileName = fullfile(ecgSignalPath, ...
        ecgSignalDirInfo(iEcgFile).name);

    TempData = load(ecgSignalFileName);
    tempDataFieldNames = fieldnames(TempData);

    % If the file contains one variable it will be taken as the ecg
    % signal.
    rawEcgSignal = TempData.(tempDataFieldNames{1}).ecgSignal;

    % Trim this signal to be 30 seconds in duration.
    rawEcgSignal = rawEcgSignal(1 : acceptableEcgLength500);

    % Initial data table. One table will be generated for each valid
    % file and stored in a seperate file in result folder.
    [noiseNames, DataTable] = generateInitialTable(nSNR, SNR, maxNosieSections);

    % Make sure the ecg signal is in column orientation and store the
    % data and file name in the data table.
    rawEcgSignal = rawEcgSignal(:);
    DataTable.ecgSignal(1, 1) = {rawEcgSignal};
    DataTable.FileName(1, 1) = ...
        ecgSignalDirInfo(iEcgFile).name(1 : end - 4);

    % Get the length of the current ecg signal.
    lengthOfThisEcgSignal = numel(rawEcgSignal);

    % Detect the r-peak locations.
    qrsLocations = TempData.(string(tempDataFieldNames)).qrsPeaks;

    % Trim this to 30 seconds of data
    qrsValidFlag = find(qrsLocations <= lengthOfThisEcgSignal);
    qrsLocations = qrsLocations(1 : numel(qrsValidFlag));

    % Convert the qrs search window to samples. This window will be used to
    % calculate the qrs amplitude.
    qrsSearchWindow = round(QRS_SEARCH_WINDOW * ecgFs);

    % Per-allocate memory.
    nQrsLocations = numel(qrsLocations);
    qrsPeakToPeak = nan(nQrsLocations, 1);

    % Calculate the peak to peak amplitude of the qrs complexes.
    for jQrsLocations = 1 : nQrsLocations

        if (qrsLocations(jQrsLocations) + qrsSearchWindow <= ...
                lengthOfThisEcgSignal) && (qrsLocations(jQrsLocations) - ...
                qrsSearchWindow >= 1)

            qrsAmpMax = max(rawEcgSignal(qrsLocations(jQrsLocations) - ...
                qrsSearchWindow : qrsLocations(jQrsLocations) + qrsSearchWindow));
            qrsAmpMin = min(rawEcgSignal(qrsLocations(jQrsLocations) - ...
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

        thisNoise = initialGenerateNoise.("EM_Noise");

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

                scaledSignal = thisNoiseSignal .* scaleFactor;
                
                % Save the noise corrupt signals in the data table.
                DataTable.(['SNR', num2str(SNR(kSNR))])(mSection, 1). ...
                    (string(noiseNames)){iGenSignal, 1} = ...
                    scaledSignal + rawEcgSignal;

            end

        end

    end

    % Call function to convert and save each signal to a readable format for
    % Deep learning models.
    convertToHDF5Format(DataTable, resultDataFolder);

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
noiseStructure.em = {};

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

% Function to recursively save data from MATLAB to H5
function convertToHDF5Format(DataTable, resultDataFolder)
% Save all signals to a more readable format

% Constants.
NUM_OF_SNR = 5;

% Assign output folders for clean and noisy signals.
outputFolderClean = fullfile(resultDataFolder, "cleanSignals");
outputFolderNoisy = fullfile(resultDataFolder, "noisySignals");

% Create folders if they dont exist.
if ~isfolder(outputFolderNoisy) mkdir(outputFolderNoisy); end
if ~isfolder(outputFolderClean) mkdir(outputFolderClean); end

% Extract the filename.
fileName = table2array(DataTable(1, 1));

% Extract the clean signal.
cleanSignal = DataTable.ecgSignal{1, 1};

% Save the clean signal in H5 format.
save(fullfile(outputFolderClean, fileName + '.h5'), 'cleanSignal');

% Now we need to loop through the noisy signals.
for iSection = 1 : height(DataTable)

    % Extract the section number 
    sectionNumber = iSection;

    % Loop through each SNR value.
    for iSNR = 1 : NUM_OF_SNR

        % Index into correspinding SNR column
        SNRValue = string(DataTable.Properties.VariableNames(:, iSNR + 3));

        % Extract the Data (nested in structs)
        SNRData = table2array(DataTable(:, iSNR + 3));

        % Grab the section specific noise structure.
        sectionNoise = SNRData(iSection).em;

        % Loop through each copy (From AR-Modelling)
        for iCopy = 1 : numel(sectionNoise)

            % Specify the noise type.
            if iCopy == 1
                noiseType = 'Original';
            else
                noiseType = string(iCopy);
            end

            % Extract the noisy ECG signal.
            noisyEcg = cell2mat(sectionNoise(iCopy));

            % Create sub directories for each SNR.
            outputFolderNoisySub = fullfile(outputFolderNoisy, SNRValue);

            % Make directory
            if ~isfolder(outputFolderNoisySub) mkdir(outputFolderNoisySub); ...
            end

            % Let's create a filename based on the type of signal.
            outputFilename = fullfile(outputFolderNoisySub, ...
                strcat(erase(fileName, '_cleanSignal'), '-', ...
                noiseType, '-', string(sectionNumber), '.h5'));

            % Save the signal in h5 format.
            save(outputFilename, 'noisyEcg');
        end
    end
end

end

%------------- END OF CODE -------------