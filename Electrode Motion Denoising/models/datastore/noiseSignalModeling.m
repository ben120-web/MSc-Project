function noiseOutputSignals = noiseSignalModeling(noiseInputSignal, fs, ...
    nOutputSignals)
% noiseSignalModeling - generates multiple copies of noise signal based on
% the model parameters estimated using the input noise signal as a
% reference. The out noisy signals are also scaled according to the SNR
% parameter.
%
% Syntax: noiseOutputSignals = noiseSignalModeling(noiseInputSignal, fs, ...
%            nOutputSignals, isNoiseSignalPowerlineNoise)
%
% Inputs:
%    noiseInputSignal - A numeric vector containing noise signal which
%       will be used as a reference to model output noise signals.
%
% Optional:
%    fs - A numeric scalar representing the sampling frequency of the
%       signals (Default: 500 Hz). [Hz]
%    nOutputSignals - A numeric scalar representing the number of
%       signals to generate/model. Output will contain nOutputSignals + 1
%       signals. The extra one coming from the original noise signal
%       (Default = 2).
%    isNoiseSignalPowerlineNoise - A logical scalar to indicate whether the
%       noisy signal is powerline noise or not (Default: false).
%
% Outputs:
%    noiseOutputSignals - A cell array containing all the noisy signals.
%
% Example:
%    noiseOutputSignals = noiseSignalModeling(noiseInputSignal);
%    noiseOutputSignals = noiseSignalModeling(noiseInputSignal, ...
%       fs, 5, true);
%
% Other m-files required: validateStandardInput.m.
% Subfunctions: none.
% MAT-files required: none.
%
%------------- BEGIN CODE --------------
%% Set constants
MAX_MODEL_ORDER = 300;
SIGNAL_LENGTH_DIVIDER = 2;
DEFAULT_FS = 500;
DEFAULT_SIGNALS_TO_GENERATE = 2;
DEFAULT_IS_NOISE_SIGNAL_POWERLINE_NOISE = false;

%% Check inputs.
% Check correct number of arguments.
minArgs = 1;
maxArgs = 4;
narginchk(minArgs, maxArgs);

% Validate noise signal.
validateStandardInput(noiseInputSignal, '1dSignal', 'noiseInputSignal', 1);

% Validate fs.
if nargin > 1 && ~isempty(fs)

    validateStandardInput(fs, 'fs', 'fs', 2);

else

    fs = DEFAULT_FS;

end


% Validate nOutputSignals.
if nargin > 2 && ~isempty(nOutputSignals)

    validateattributes(nOutputSignals, {'numeric'}, {'scalar', 'finite', ...
        'real', 'positive', 'integer'}, mfilename, 'nOutputSignals', 3);

else

    nOutputSignals = DEFAULT_SIGNALS_TO_GENERATE;

end

% Get the length of the signal and pre-allocate memory.
lengthOfNoiseSignal = numel(noiseInputSignal);
noiseOutputSignals = cell(nOutputSignals + 1, 1);

% Use the noiseInputSignal to model different cases of noisy signals with
% similar PSD.
modelOrder = min([MAX_MODEL_ORDER, lengthOfNoiseSignal / SIGNAL_LENGTH_DIVIDER]);

noiseSignalMean = mean(noiseInputSignal);
unbiasedInputSignal = noiseInputSignal - noiseSignalMean;

% Estimate the model coefficients.
[modelCoefs, modelVariance] = arburg(unbiasedInputSignal, modelOrder);

% Set the random number generator and generate the input signals for the
% model.
rng(45, 'twister');
inputSignalForModel = randn(lengthOfNoiseSignal, nOutputSignals);

% Get the initial state for the model.
zi = filtic(1, modelCoefs, flipud(unbiasedInputSignal));

% Estimate the signals using the model parameters.
estimatedSignals = noiseSignalMean + (filter(1, modelCoefs, ...
    inputSignalForModel, zi)) .* sqrt(modelVariance);

% Add the initial input noise signal to the output matirx.
noiseOutputSignals{1, 1} = noiseInputSignal;

for iOutputSignal = 2 : nOutputSignals + 1
    
    noiseOutputSignals{iOutputSignal, 1} = estimatedSignals(:, iOutputSignal - 1);

end

end
%------------- END OF CODE -------------