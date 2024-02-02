function createSyntheticCleanEcgSignals()

% Define paraters
samplingFrequency = 500; % Always work at 500Hz.

HR_TO_GENERATE = [50, 60, 70, 80, 90, 100];
MIN_ANGLES_OF_EXTREMA = [-90 -35 -20 -5 80];
MAX_ANGLES_OF_EXTRAMA = [-50 5 20 35 120];
MIN_Z_POSITION_OF_EXTRAMA = [0 -10 5 -15 0];
MAX_Z_POSITION_OF_EXTRAMA = [3 30 55 0 5];
MIN_GAUSSIAN_WIDTH = [0.1 0.05 0.05 0.05 0.2];
MAX_GAUSSIAN_WIDTH = [0.4 0.15 0.15 0.15 0.6];

% Number of different sigals to generate.
numOfCleanSignals = 10000;

% Employ Latin Hyper Cube sampling to generate all possible combinations of
% parameter values.
[anglesOfExtrema, ~] = latinHyperCubeSampling(numOfCleanSignals, MIN_ANGLES_OF_EXTREMA, MAX_ANGLES_OF_EXTRAMA);
[zPositionOfExtrema, ~] = latinHyperCubeSampling(numOfCleanSignals, MIN_Z_POSITION_OF_EXTRAMA, MAX_Z_POSITION_OF_EXTRAMA);
[gaussianWidth, ~] = latinHyperCuDefbeSampling(numOfCleanSignals, MIN_GAUSSIAN_WIDTH, MAX_GAUSSIAN_WIDTH);

% Number of heartrates
numOfHeartRates = numel(HR_TO_GENERATE);

% Define output folder
outputFolder = erase(mfilename('fullpath'), mfilename);
outputFolder = fullfile(outputFolder, ...
    'cleanSignals');

% Make directory
if ~isfolder(outputFolder) mkdir(outputFolder); end

% Initialise Structure
signalData = struct();

% Loop through each parameter setting.
for iHeartRate = 1 : numOfHeartRates

    meanHr = HR_TO_GENERATE(iHeartRate);

    % Loop through each signal to generate
    for iSignal = 1 : numOfCleanSignals

        % Extract data.
        angleData = anglesOfExtrema(iSignal, :);
        zData = zPositionOfExtrema(iSignal, :);
        widthData = gaussianWidth(iSignal, :);

        % Call ECGSYN MATLAB function to generate signal.
        try
            [cleanEcgSignal, qrsLocations] = ecgsyn(samplingFrequency, 256, 0, meanHr, ...
                1, 0.5, samplingFrequency, angleData, zData, ...
                widthData);
            
            % Save all peak locations
            peakLocations = find(qrsLocations == 3);

            % Create a structure containing data
            signalData.ecgSignal = cleanEcgSignal;
            signalData.qrsPeaks = peakLocations;

        catch
            continue
        end

        % Save the outputs with appropriate file names.
        fileName = fullfile(string(meanHr) + "BPM_" + string(iSignal) + "_cleanSignal");
        save(fullfile(outputFolder, fileName), "signalData")

    end

    % Save parameters
    save(fullfile(outputFolder, "AngleData"), 'anglesOfExtrema');
    save(fullfile(outputFolder, "zData"), 'zPositionOfExtrema');
    save(fullfile(outputFolder, "widthData"), 'gaussianWidth');
end
end

function [X_scaled,X_normalized] = latinHyperCubeSampling(n, min_ranges_p, max_ranges_p)
%lhsdesign_modified is a modification of the Matlab Statistics function lhsdesign.
%It might be a good idea to jump straight to the example to see what does
%this function do.
%The following is the description of lhsdesign from Mathworks documentation
% X = lhsdesign(n,p) returns an n-by-p matrix, X, containing a latin hypercube sample of n values on each of p variables.
%For each column of X, the n values are randomly distributed with one from each interval (0,1/n), (1/n,2/n), ..., (1-1/n,1), and they are randomly permuted.
%lhsdesign_modified provides a latin hypercube sample of n values of
%each of p variables but unlike lhsdesign, the variables can range between
%any minimum and maximum number specified by the user, where as lhsdesign
%only provide data between 0 and 1 which might not be very helpful in many
%practical problems where the range is not bound to 0 and 1
%
%Inputs: 
%       n: number of radomly generated data points
%       min_ranges_p: [1xp] or [px1] vector that contains p values that correspond to the minimum value of each variable
%       max_ranges_p: [1xp] or [px1] vector that contains p values that correspond to the maximum value of each variable
%Outputs
%       X_scaled: [nxp] matrix of randomly generated variables within the
%       min/max range that the user specified
%       X_normalized: [nxp] matrix of randomly generated variables within the
%       0/1 range 
%
%Example Usage: 
%       [X_scaled,X_normalized]=lhsdesign_modified(100,[-50 100 ],[20  300]);
%       figure
%       subplot(2,1,1),plot(X_scaled(:,1),X_scaled(:,2),'*')
%       title('Random Variables')
%       xlabel('X1')
%       ylabel('X2')
%       grid on
%       subplot(2,1,2),plot(X_normalized(:,1),X_normalized(:,2),'r*')
%       title('Normalized Random Variables')
%       xlabel('Normalized X1')
%       ylabel('Normalized X2')
%       grid on
p=length(min_ranges_p);
[M,N]=size(min_ranges_p);
if M<N
    min_ranges_p=min_ranges_p';
end
    
[M,N]=size(max_ranges_p);
if M<N
    max_ranges_p=max_ranges_p';
end
slope=max_ranges_p-min_ranges_p;
offset=min_ranges_p;
SLOPE=ones(n,p);
OFFSET=ones(n,p);
for i=1:p
    SLOPE(:,i)=ones(n,1).*slope(i);
    OFFSET(:,i)=ones(n,1).*offset(i);
end
X_normalized = lhsdesign(n,p);
X_scaled=SLOPE.*X_normalized+OFFSET;

end