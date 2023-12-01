function createSyntheticCleanEcgSignals()

% Define paraters
samplingFrequency = 500; % Always work at 500Hz.

HR_TO_GENERATE = [50, 60, 70, 80, 90, 100];
FEATURES = ["P", "Q", "R", "S", "T"];
MIN_ANGLES_OF_EXTREMA = [-90 -35 -20 -5 80];
MAX_ANGLES_OF_EXTRAMA = [-50 5 20 35 120];
MIN_Z_POSITION_OF_EXTRAMA = [0 -10 5 -15 0];
MAX_Z_POSITION_OF_EXTRAMA = [3 30 55 0 5];
MIN_GAUSSIAN_WIDTH = [0.25 0.1 0.1 0.1 0.4];
MAX_GAUSSIAN_WIDTH = [0.25 0.1 0.1 0.1 0.4];

% Create a structure with all features to change.
featureStruct = struct();
featureStruct.extremaAngles(:, 1) = MIN_ANGLES_OF_EXTREMA';
featureStruct.extremaAngles(:, 2) = MAX_ANGLES_OF_EXTRAMA';
featureStruct.zPosition(:, 1) = MIN_Z_POSITION_OF_EXTRAMA';
featureStruct.zPosition(:, 2) = MAX_Z_POSITION_OF_EXTRAMA';
featureStruct.width(:, 1) = MIN_GAUSSIAN_WIDTH';
featureStruct.width(:, 2) = MAX_GAUSSIAN_WIDTH';


% Loop through each parameter setting.
for iHeartRate = 1 : numOfHeartRates

    meanHr = HR_TO_GENERATE(iHeartRate);

    % State number of features we want to iterate through.
    nFeatures = numel(fieldnames(featureStruct));

    % Few more loops to cover each shape.
    for iFeature = 1 : nFeatures

        featureNames = fieldnames(featureStruct);

        % Specify feature
        feature = string(featureNames(iFeature));

        % Grab Data
        data = featureStruct.(feature);

        minValues = data(:, 1);
        maxValues = data(:, 2);

        % Get number of morhologies to iterate through
        numOfMorphologies = height(data);

        % Iterate through specific feature
        for iMorphology = 1 : numOfMorphologies

            

            % Call ECGSYN MATLAB function to generate signal.
            [ecgSignal, peakLocations] = ecgsyn(samplingFrequency, 256, 0, meanHr, ...
                1, 0.5, samplingFrequency, [-70 -15 0 15 100], [1.2 -5 30 -7.5 0.75], ...
                    [0.25 0.1 0.1 0.1 0.4]);

            % Save the outputs with appropriate file names.
            fileName = fullfile(string(meanHr) + "BPM" + "cleanSignal");
        end
    end
end