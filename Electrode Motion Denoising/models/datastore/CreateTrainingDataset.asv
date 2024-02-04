function CreateTrainingDataset()
% Script CreateTrainingDataset.m will generate the dataset used for
% training various learning algorithms on removing EM noise.

%% Main processing

% Call function to generate reference noisy signals.
extractAndPreProcessNoiseSignal(); % These get saved in datastore/noiseSignal

% Call function to syntetically generate Clean ECG signals.
createSyntheticCleanEcgSignals();

% Define inputs.
noiseSignalPath = fullfile(erase(mfilename('fullpath'), mfilename), "noiseSignal");
ecgSignalPath = fullfile(erase(mfilename('fullpath'), mfilename), "cleanSignals"); % Create a database of clean ECG // Ensure 500Hz. Keep in .mat format.

AverageEcgLength = 30; % Keep as constants.
maxNosieSections = 10; % Can vary this.
SNR = [0 6 12 18 24];
numberOfGeneratedNoisySignals = 10; % Can vary this.

% Generate noisy database
generatingNoisyEcgDatabase(noiseSignalPath, ...
    ecgSignalPath, 500, AverageEcgLength, maxNosieSections, SNR, ...
    numberOfGeneratedNoisySignals);


end
