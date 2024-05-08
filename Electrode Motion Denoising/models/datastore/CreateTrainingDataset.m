function CreateTrainingDataset()
% Script CreateTrainingDataset.m will generate the dataset used for
% training Machine / Deep Learning algorithms. It will create a dataset
% with clean reference signals, as well as a range of signals corrupt with
% various amounts of electrode motion noise. The idea is that the reference
% signals will act as a ground truth, and algorithms will learn what
% electrode motion looks like (Both temperal and frequency
% characteristics).
%
% Step 1: ExctractAndPreProcessNoiseSignal - Extracts the Electrode Motion
% noise file from Physionet. This noise file has been collected through
% specialised placement of electrodes.
%
% Step 2: createSyntheticCleanEcgSignals - Generates synthetic ECG signals
% based on 1st order differential equations. Sampling is used to generate
% signals with a range of morphologies.
%
% Step 3: generateNoisyDatabase - Corrupts the clean ECG signals with
% variuous amounts of electrode motion noise. 

%% Main processing

% Call function to generate reference noisy signals.
extractAndPreProcessNoiseSignal(); % These get saved in datastore/noiseSignal

% Call function to syntetically generate Clean ECG signals.
createSyntheticCleanEcgSignals();

% Define inputs.
noiseSignalPath = fullfile(erase(mfilename('fullpath'),...
    mfilename), "noiseSignal");
ecgSignalPath = fullfile(erase(mfilename('fullpath'),...
    mfilename), "cleanSignals"); % Create a database of clean ECG // Ensure 500Hz. Keep in .mat format.

% Define input parameters.
maxNosieSections = 10; % Can vary this.
SNR = [0 6 12 18 24];
numberOfGeneratedNoisySignals = 10; % Can vary this.

% Generate noisy database
generatingNoisyEcgDatabase(noiseSignalPath, ...
    ecgSignalPath, 500, maxNosieSections, SNR, ...
    numberOfGeneratedNoisySignals);


end
% ------------------- END OF CODE ----------------------