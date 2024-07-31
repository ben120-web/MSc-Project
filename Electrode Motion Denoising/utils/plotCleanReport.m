% Define the base directory where the files are located
baseDir = 'C:\B-Secur\MSc Project\ElectrodeMotionionDenoisingFramework\Electrode Motion Denoising\models\datastore\cleanSignals'; % Update this with the actual path

% BPM values and corresponding filenames
BPM_values = 50:10:100;
numSignals = [1, 3, 4, 5];

% Create a 5x4 tiled layout for plotting
figure;
tiledlayout(5, 4);

% Loop through each BPM value
for i = 1:length(BPM_values)
    BPM = BPM_values(i);
    
    % Loop through each signal number for the current BPM, skipping number 2
    for j = 1:length(numSignals)
        signalNum = numSignals(j);
        
        % Construct the filename
        filename = fullfile(baseDir, sprintf('%dBPM_%d_cleanSignal.mat', BPM, signalNum));
        
        % Load the signal data
        data = load(filename);
        
        % Assume the signal data is stored in a variable named 'cleanSignal'
        signal = data.signalData.ecgSignal;

        signal = signal(1000 : 5000);
        
        % Determine the current tile position
        tileIndex = (i-1)*length(numSignals) + j;
        
        % Plot the signal in the appropriate tile
        nexttile(tileIndex);
        plot(signal);
        title(sprintf('%d BPM - Signal %d', BPM, signalNum));
        xlabel('Time');
        ylabel('Amplitude');
        xlim([0, 4000])
    end
end
