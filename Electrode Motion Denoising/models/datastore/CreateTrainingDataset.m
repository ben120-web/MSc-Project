function CreateTrainingDataset()
% Script CreateTrainingDataset.m will generate the dataset used for
% training various learning algorithms on removing EM noise.

%% Main processing

% Call function to generate reference noisy signals.
emNoise = ...
    extractAndPreProcessNoiseSignal(); % These get saved in datastore/noiseSignal

% Define inputs.
noiseSignalPath = fullfile(erase(mfilename('fullpath'), mfilename), "noiseSignal");
ecgSignalPath = ; % Create a database of clean ECG // Ensure 500Hz. Keep in .mat format.

AverageEcgLength = ; % Keep as constants.
maxNosieSections = 10; % Can vary this.
SNR = [0 6 12 18 24];
numberOfGeneratedNoisySignals = 10; % Can vary this.


% Generate noisy database
generatingNoisyEcgDatabase(noiseSignalPath, ...
    ecgSignalPath, 500, AverageEcgLength, maxNosieSections, SNR, ...
    numberOfGeneratedNoisySignals);



