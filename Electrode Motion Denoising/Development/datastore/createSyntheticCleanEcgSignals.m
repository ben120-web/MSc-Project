function createSyntheticCleanEcgSignals()
% createSyntheticCleanEcgSignals - generates a database of clean ECG
% signals with different feature morphologies. Saves these in a unique
% directory.
%
% Syntax: createSyntheticCleanEcgSignals()
%
% Inputs:
%    None Required.
%
% Outputs: none.
%
% Other m-files required: none.
% Subfunctions: none.
% MAT-files required: generic-features-noise-sources.mat.

%------------- BEGIN CODE --------------

% Define paraters
samplingFrequency = 500; % Always work at 500Hz.

HR_TO_GENERATE = [50, 60, 70, 80, 90, 100];
MIN_ANGLES_OF_EXTREMA = [-60 -12 0 12 80];
MAX_ANGLES_OF_EXTRAMA = [-80 -25 0 25 120];
MIN_Z_POSITION_OF_EXTRAMA = [0.9 -7 20 -10 0.5];
MAX_Z_POSITION_OF_EXTRAMA = [1.5 -2 45 -5 1];
MIN_GAUSSIAN_WIDTH = [0.25 0.1 0.1 0.1 0.4];
MAX_GAUSSIAN_WIDTH = [0.75 0.4 0.8 0.1 0.4];

% Number of different sigals to generate.
numOfCleanSignals = 5;

% Employ Latin Hyper Cube sampling to generate all possible combinations of
% parameter values.
[anglesOfExtrema, ~] = latinHyperCubeSampling(numOfCleanSignals, MIN_ANGLES_OF_EXTREMA, MAX_ANGLES_OF_EXTRAMA);
[zPositionOfExtrema, ~] = latinHyperCubeSampling(numOfCleanSignals, MIN_Z_POSITION_OF_EXTRAMA, MAX_Z_POSITION_OF_EXTRAMA);
[gaussianWidth, ~] = latinHyperCubeSampling(numOfCleanSignals, MIN_GAUSSIAN_WIDTH, MAX_GAUSSIAN_WIDTH);

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
            cleanEcgSignal = ecgsyn(samplingFrequency, 256, 0, meanHr, ...
                1, 0.5, samplingFrequency, angleData, zData, ...
                widthData);

            % Get QRS location
            [~, qrsLocations] = findpeaks(cleanEcgSignal, 'MinPeakHeight', ...
                mean(cleanEcgSignal) + 0.5 * std(cleanEcgSignal), ...
                'MinPeakDistance', round(0.6 * samplingFrequency));

            % Apply filter to remove baseline drift.
            cleanEcgSignal = remove_baseline_drift(cleanEcgSignal, samplingFrequency);

            % Call function to validate the ECG is realistic
            signalValid = validateEcgIsRealistic(cleanEcgSignal, ...
                qrsLocations, samplingFrequency);

            % If the signal is not valid, continue loop
            if ~signalValid
                continue
            end

            % Create a structure containing data
            signalData.ecgSignal = cleanEcgSignal;
            signalData.qrsPeaks = qrsLocations;

        catch
            continue
        end

        % Save the outputs with appropriate file names
        fileName = fullfile(outputFolder, string(meanHr) + "BPM_" + string(iSignal) + "_cleanSignal");
        save(fileName, "signalData")

    end

    % Save parameters
    save(fullfile(outputFolder, "AngleData"), 'anglesOfExtrema');
    save(fullfile(outputFolder, "zData"), 'zPositionOfExtrema');
    save(fullfile(outputFolder, "widthData"), 'gaussianWidth');
end

end

function [X_scaled, X_normalized] = latinHyperCubeSampling(n, min_ranges_p, max_ranges_p)
% Perform Latin Hypercube sampling.
    p = length(min_ranges_p);
    [M, N] = size(min_ranges_p);
    if M < N
        min_ranges_p = min_ranges_p';
    end
    
    [M, N] = size(max_ranges_p);
    if M < N
        max_ranges_p = max_ranges_p';
    end
    slope = max_ranges_p - min_ranges_p;
    offset = min_ranges_p;
    SLOPE = ones(n, p);
    OFFSET = ones(n, p);
    for i = 1:p
        SLOPE(:, i) = ones(n, 1) .* slope(i);
        OFFSET(:, i) = ones(n, 1) .* offset(i);
    end
    X_normalized = lhsdesign(n, p);
    X_scaled = SLOPE .* X_normalized + OFFSET;
end

function signalValid = validateEcgIsRealistic(ecgSignal, peakLocations, samplingFrequency)
    % Function to validate if the morphologies of the ECG signal is realistic,
    % if deemed unrealistic, the signal is scrapped from the database. Realism
    % is determined from set physiological limits of each feature within the
    % ECG signal.

    % Constants
    P_WAVE_DURATION = [0.02, 0.4];
    QRS_DURATION = [0.01, 0.3];
    T_WAVE_DURATION = [0.08, 0.5];
    P_WAVE_AMPLITUDE = [0.05, 0.5];
    QRS_AMPLITUDE = [0.06, 2];
    T_WAVE_AMPLITUDE = [0.01, 0.9];
    QT_INTERVAL = [0.15, 0.66];

    fiducials = extract_ecg_fiducials(ecgSignal, peakLocations, samplingFrequency);

    % Since all beats are the same, we only need to validate on 1.
    pOnset = fiducials.P_onset(1) - 10;
    pPeak = fiducials.P_peak(1);
    pOffset = fiducials.P_offset(1) + 10;
    qOnset = fiducials.QRS_onset(1) - 10;
    rPeak = fiducials.QRS_peak(1);
    qOffset = fiducials.QRS_offset(1) + 10;
    tOnset = fiducials.T_onset(1) - 10;
    tPeak = fiducials.T_peak(1);
    tOffset = fiducials.T_offset(1) + 10;

    % Lets calculate the features we need to check. (convert to s and mV)
    pWaveDuration = abs(pOffset - pOnset) / samplingFrequency;
    QRSDuration = abs(qOffset - qOnset) / samplingFrequency;
    tWaveDuration = abs(tOffset - tOnset) / samplingFrequency;
    pWaveAmplitude = abs(ecgSignal(pPeak));
    qrsAmplitude = abs(ecgSignal(rPeak));
    tWaveAmplitude = abs(ecgSignal(tPeak));
    qtInterval = abs(tOffset - qOnset) / samplingFrequency;

    % Perform validation check. Only pass a signal that meets all requirements.
    if pWaveDuration <= P_WAVE_DURATION(2) && pWaveDuration >= P_WAVE_DURATION(1) ...
            && QRSDuration <= QRS_DURATION(2) && QRSDuration >= QRS_DURATION(1) ...
            && tWaveDuration <= T_WAVE_DURATION(2) && tWaveDuration >= T_WAVE_DURATION(1) ...
            && pWaveAmplitude <= P_WAVE_AMPLITUDE(2) && pWaveAmplitude >= P_WAVE_AMPLITUDE(1) ...
            && qrsAmplitude <= QRS_AMPLITUDE(2) && qrsAmplitude >= QRS_AMPLITUDE(1) ...
            && tWaveAmplitude <= T_WAVE_AMPLITUDE(2) && tWaveAmplitude >= T_WAVE_AMPLITUDE(1) ...
            && qtInterval <= QT_INTERVAL(2) && qtInterval >= QT_INTERVAL(1)

        % Set flag as valid.
        signalValid = true;
    else
        signalValid = false;
    end
end

function fiducials = extract_ecg_fiducials(ecgSignal, peakLocations, samplingRate)
    % Function to extract fiducial points (onset, peak, offset) for P wave,
    % QRS complex, and T wave from an ECG signal.
    % INPUTS:
    % ecgSignal: Array containing the ECG signal.
    % peakLocations: Array of detected R-peak locations.
    % samplingRate: Sampling rate of the ECG signal.
    % OUTPUT:
    % fiducials: Structure containing the onset, peak, and offset of P wave,
    % QRS complex, and T wave.

    % Initialize structure to hold fiducials
    fiducials = struct('P_onset', [], 'P_peak', [], 'P_offset', [], ...
        'QRS_onset', [], 'QRS_peak', [], 'QRS_offset', [], ...
        'T_onset', [], 'T_peak', [], 'T_offset', []);

    % Bandpass filter the ECG signal to remove baseline wander and noise
    ecgFiltered = bandpass(ecgSignal, [0.5 45], samplingRate);

    R_locs = peakLocations;

    % Loop through each detected R peak to find fiducial points
    for iPeak = 1:length(R_locs)
        % Skip first and last peaks to avoid boundary issues
        if iPeak == 1 || iPeak == length(R_locs)
            continue
        end

        % Define search windows relative to R peak location
        win_P = max(1, R_locs(iPeak) - round(0.3 * samplingRate)) ...
            : R_locs(iPeak) - round(0.1 * samplingRate);

        win_QRS = max(1, R_locs(iPeak) - round(0.1 * samplingRate)):min(length(ecgSignal), R_locs(iPeak) + round(0.1 * samplingRate));
        win_T = R_locs(iPeak) + round(0.1 * samplingRate):min(length(ecgSignal), R_locs(iPeak) + round(0.4 * samplingRate));

        % P wave detection
        [~, P_peak_idx] = max(ecgFiltered(win_P));
        P_peak = win_P(P_peak_idx);
        P_onset = find_onset(ecgFiltered, P_peak, -1, samplingRate);
        P_offset = find_offset(ecgFiltered, P_peak, 1, samplingRate);

        % QRS complex detection using Pan-Tompkins algorithm
        [~, QRS_peak_idx] = max(ecgFiltered(win_QRS));
        QRS_peak = win_QRS(QRS_peak_idx);
        QRS_onset = find_onset(ecgFiltered, QRS_peak, -1, samplingRate);
        QRS_offset = find_offset(ecgFiltered, QRS_peak, 1, samplingRate);

        % T wave detection
        [~, T_peak_idx] = max(ecgFiltered(win_T));
        T_peak = win_T(T_peak_idx);
        T_onset = find_onset(ecgFiltered, T_peak, -1, samplingRate);
        T_offset = find_offset(ecgFiltered, T_peak, 1, samplingRate);

        % Store fiducials
        fiducials.P_onset = [fiducials.P_onset, P_onset];
        fiducials.P_peak = [fiducials.P_peak, P_peak];
        fiducials.P_offset = [fiducials.P_offset, P_offset];
        fiducials.QRS_onset = [fiducials.QRS_onset, QRS_onset];
        fiducials.QRS_peak = [fiducials.QRS_peak, QRS_peak];
        fiducials.QRS_offset = [fiducials.QRS_offset, QRS_offset];
        fiducials.T_onset = [fiducials.T_onset, T_onset];
        fiducials.T_peak = [fiducials.T_peak, T_peak];
        fiducials.T_offset = [fiducials.T_offset, T_offset];
    end
end

function onset = find_onset(ecgFiltered, peak, direction, samplingRate)
    % Helper function to find onset of a wave
    window = round(0.1 * samplingRate); % 100 ms window
    idx = peak;
    baseline = mean(ecgFiltered(max(1, idx-window):min(length(ecgFiltered), idx+window)));
    threshold = baseline + 0.05 * (ecgFiltered(peak) - baseline);

    if direction == -1
        while idx > 1
            if ecgFiltered(idx) < threshold
                break
            end
            idx = idx + direction;
        end
    else
        while idx < length(ecgFiltered)
            if ecgFiltered(idx) < threshold
                break
            end
            idx = idx + direction;
        end
    end

    onset = idx;
end

function offset = find_offset(ecgFiltered, peak, direction, samplingRate)
    % Helper function to find offset of a wave
    window = round(0.1 * samplingRate); % 100 ms window
    idx = peak;
    baseline = mean(ecgFiltered(max(1, idx-window):min(length(ecgFiltered), idx+window)));
    threshold = baseline + 0.05 * (ecgFiltered(peak) - baseline);

    if direction == -1
        while idx > 1
            if ecgFiltered(idx) < threshold
                break
            end
            idx = idx + direction;
        end
    else
        while idx < length(ecgFiltered)
            if ecgFiltered(idx) < threshold
                break
            end
            idx = idx + direction;
        end
    end

    offset = idx;
end

function ecgFiltered = remove_baseline_drift(ecgSignal, samplingRate)
    % remove_baseline_drift - Removes baseline drift from ECG signal
    %
    % Syntax: ecgFiltered = remove_baseline_drift(ecgSignal, samplingRate)
    %
    % Inputs:
    %    ecgSignal - Array containing the ECG signal
    %    samplingRate - Sampling rate of the ECG signal
    %
    % Outputs:
    %    ecgFiltered - Baseline drift removed ECG signal

    % Define the cutoff frequency for the high-pass filter
    cutoffFreq = 0.5; % 0.5 Hz cutoff frequency to remove baseline drift

    % Design the high-pass filter using a Butterworth filter
    [b, a] = butter(2, cutoffFreq / (samplingRate / 2), 'high');

    % Apply the high-pass filter to the ECG signal
    ecgFiltered = filtfilt(b, a, ecgSignal);
end

% ---------------------- END OF CODE ------------------------------
