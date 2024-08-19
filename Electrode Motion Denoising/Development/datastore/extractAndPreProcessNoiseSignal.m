function emNoise = ...
    extractAndPreProcessNoiseSignal()
% referenceNoiseSignalGeneration - reads the noise signals, from Physionet,
% which are stored in the ADF process them as done for signal quality
% assessment report and save them in datastore folder. 
% To generate multiple instances of noise with noiseSignalModeling
% function the noise signals generated in this files should be used.
%
% Syntax: [bwNoise, emNoise, maNoise, powerlineNoise50, powerlineNoise60] = ...
%            referenceNoiseSignalGeneration(PowerLineOpts)
%
% Inputs:
%    None Required.
%
% Outputs:
%    emNoise - A numeric vector containing the electrode motion noise signal.
% Other m-files required: none.
% Subfunctions: none.
% MAT-files required: generic-features-noise-sources.mat.

%------------- BEGIN CODE --------------
%% Set constants
SIGNAL_FS = 500; % [Hz]
FILTER_ORDER = 20;
FILTER_ATTENUATION_DB = 60; % [dB]
SIGNAL_FN = SIGNAL_FS / 2; % [Hz]
BASELINE_MAX_FREQ = 1; % [Hz]
ELECTRODE_MOTION_MAX_FREQ = 50; % [Hz]

%% Check inputs.
% Check correct number of arguments.
minArgs = 0;
maxArgs = 1;
narginchk(minArgs, maxArgs)

dataPath = erase(mfilename('fullpath'), mfilename);

% Load the stored noise signals to a temporary structure.
TempStruct = load(fullfile(dataPath, 'generic-features-noise-sources.mat'));

% Extract required variables and resample to 500Hz. (500Hz will be standard
% samkpling frequency in this work)
fsNoise = TempStruct.fs;
emNoise = resample(TempStruct.em(:, 1), 500, fsNoise);

%% Filter the signals to ensure frequencies do not overlap.

% % Band pass electrode motion to ensure it doesn't contain any baseline
% % wander or muscle noise.
[zEM, pEM, kEM] = cheby2(FILTER_ORDER, FILTER_ATTENUATION_DB, ...
    [BASELINE_MAX_FREQ, ELECTRODE_MOTION_MAX_FREQ] / SIGNAL_FN, 'bandpass');
[sosEM, gEM] = zp2sos(zEM, pEM, kEM);
emNoise = filtfilt(sosEM, gEM, emNoise);

outputFolderName = fullfile(dataPath, ...
    'noiseSignal');

if ~isfolder(outputFolderName)

    mkdir(outputFolderName);

end

outPutFileName = fullfile(outputFolderName, 'noiseSignal.mat');
save(outPutFileName, 'emNoise', 'SIGNAL_FS');

end

%------------- END OF CODE -------------