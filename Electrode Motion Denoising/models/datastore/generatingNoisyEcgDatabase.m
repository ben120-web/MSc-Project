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

% Read the path to the client secret from environment variable
PATH_TO_TOKEN = "C:\B-Secur\temp\secret\" + ...
    "client_secret_327364716932-78oacfgib4ilrotdphikdpbkvfnlk76c." + ...
    "apps.googleusercontent.com.json";

% Get the information from the noise signals folder.
noiseSignalDirInfo = dir(fullfile(noiseSignalPath, '*.mat'));
ecgSignalDirInfo = dir(fullfile(ecgSignalPath, '*.mat'));

% Read stored noise signals from mat file.
initialNoiseData = load(fullfile(noiseSignalPath, noiseSignalDirInfo.name));

% Calculate the minimum and maximum length of ecg signals which will be
% used for noise generation.
acceptableEcgLength500 = ECG_LENGTH_SECONDS * ecgFs;

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

if ~isfolder(resultDataFolder); mkdir(resultDataFolder); end % Make directory.

% Filter out meta data 
filterFlag = string({ecgSignalDirInfo.name}) == "AngleData.mat" | ...
    string({ecgSignalDirInfo.name}) == "widthData.mat" | ...
    string({ecgSignalDirInfo.name}) == "zData.mat";

ecgSignalDirInfo(filterFlag) = [];

% Number of valid ecg files.
nEcgFiles = numel(ecgSignalDirInfo);

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
    % file and stored in a separate file in result folder.
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

    % Pre-allocate memory.
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
    
    % Get the access token for google drive.
    access_token = loadGoogleDrive(PATH_TO_TOKEN);

    % Call function to convert and save each signal to a readable format for
    % Deep learning models.
    convertToHDF5Format(DataTable, resultDataFolder, access_token);

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

function convertToHDF5Format(DataTable, resultDataFolder, accessToken)
% Save all signals to a more readable format

% Constants.
NUM_OF_SNR = 5;
UPLOAD_URL = 'https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart';
BOUNDARY = '----WebKitFormBoundary7MA4YWxkTrZu0gW';

% Assign output folders for clean and noisy signals.
outputFolderClean = fullfile(resultDataFolder, "cleanSignals");
outputFolderNoisy = fullfile(resultDataFolder, "noisySignals");

% Create folders if they don't exist.
if ~isfolder(outputFolderNoisy); mkdir(outputFolderNoisy); end
if ~isfolder(outputFolderClean); mkdir(outputFolderClean); end

% Extract the filename.
fileName = table2array(DataTable(1, 1));

% Extract the clean signal.
cleanSignal = DataTable.ecgSignal{1, 1};

% Save the clean signal in H5 format.
cleanSignalPath = fullfile(outputFolderClean, fileName + '.h5');
saveHDF5(cleanSignalPath, cleanSignal);

% Upload clean signal to Google Drive
uploadToDrive(cleanSignalPath, accessToken, UPLOAD_URL, BOUNDARY, getFolderId('cleanSignals', accessToken));

% Now we need to loop through the noisy signals.
for iSection = 1 : height(DataTable)

    % Extract the section number
    sectionNumber = iSection;

    % Loop through each SNR value.
    for iSNR = 1 : NUM_OF_SNR

        % Index into corresponding SNR column
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
            if ~isfolder(outputFolderNoisySub); mkdir(outputFolderNoisySub); end

            % Create the filename based on the type of signal.
            outputFilename = fullfile(outputFolderNoisySub, ...
                strcat(erase(fileName, '_cleanSignal'), '-', ...
                noiseType, '-', string(sectionNumber), '.h5'));

            % Save the noisy signal in H5 format.
            saveHDF5(outputFilename, noisyEcg);

            % Upload noisy signal to Google Drive
            uploadToDrive(outputFilename, accessToken, UPLOAD_URL, BOUNDARY, getFolderId(['noisySignals/' SNRValue], accessToken));
        end
    end
end

end

% Function to save data in HDF5 format
function saveHDF5(filename, data)
    % Check if the file already exists and delete it
    if isfile(filename)
        delete(filename);
    end

    % Create the HDF5 file and dataset
    h5create(filename, '/data', size(data));
    h5write(filename, '/data', data);
end

% Function to upload file to Google Drive
function uploadToDrive(filePath, accessToken, UPLOAD_URL, BOUNDARY, parentFolder)
    % Read the file content
    fileContent = fileread(filePath);

    % Extract the filename
    [~, fileName, ext] = fileparts(filePath);
    fileName = strcat(fileName, ext);

    % Create metadata part for google drive upload.
    metadata = struct('name', fileName, 'parents', {parentFolder});
    metadataJson = jsonencode(metadata);

    % Create the multipart body
    body = sprintf([...
        '--%s\n' ...
        'Content-Type: application/json; charset=UTF-8\n\n' ...
        '%s\n' ...
        '--%s\n' ...
        'Content-Type: application/octet-stream\n\n' ...
        '%s\n' ...
        '--%s--\n'], ...
        BOUNDARY, metadataJson, BOUNDARY, fileContent, BOUNDARY);

    % Set HTTP headers
    headers = [...
        matlab.net.http.HeaderField('Authorization', ['Bearer ' accessToken]), ...
        matlab.net.http.HeaderField('Content-Type', ['multipart/related; boundary="' BOUNDARY '"'])];

    % Create HTTP request
    request = matlab.net.http.RequestMessage('post', headers, matlab.net.http.MessageBody(body));

    % Send HTTP request
    response = request.send(UPLOAD_URL);

    % Display response
    disp(response.Body.Data);

    % Check for errors in the response
    if isfield(response.Body.Data, 'error')
        error('Google Drive API error: %s', response.Body.Data.error.message);
    end
end

function folderId = getFolderId(folderPath, accessToken)
% Get the folder ID for a given folder path, creating folders if necessary

% Define the base URL for Google Drive API
driveApiUrl = 'https://www.googleapis.com/drive/v3/files';

% Ensure folderPath is a string scalar
folderPath = convertCharsToStrings(folderPath);

% Split the folder path into individual folder names
folderParts = strsplit(folderPath, '/');

% Initialize the parent ID (root folder)
parentId = 'root';

% Loop through each part of the folder path
for i = 1:numel(folderParts)
    % Search for the current folder in the parent folder
    searchUrl = [driveApiUrl, '?q=', ...
        'name = "', folderParts{i}, '" and ', ...
        '"', parentId, '" in parents and ', ...
        'mimeType = "application/vnd.google-apps.folder" and ', ...
        'trashed = false', ...
        '&fields=files(id)&spaces=drive'];

    % Set the HTTP options
    options = weboptions('HeaderFields', {'Authorization', ['Bearer ' accessToken]});

    % Send the search request
    response = webread(searchUrl, options);

    % Check if the folder exists
    if isempty(response.files)
        % If the folder does not exist, create it
        createUrl = driveApiUrl;
        createBody = jsonencode(struct(...
            'name', folderParts{i}, ...
            'mimeType', 'application/vnd.google-apps.folder', ...
            'parents', {parentId}));

        % Set the HTTP headers for the create request
        headers = [...
            matlab.net.http.HeaderField('Authorization', ['Bearer ' accessToken]), ...
            matlab.net.http.HeaderField('Content-Type', 'application/json')];

        % Send the create request
        request = matlab.net.http.RequestMessage('POST', headers, matlab.net.http.MessageBody(createBody));
        response = request.send(createUrl);

        % Get the ID of the newly created folder
        parentId = response.Body.Data.id;
    else
        % If the folder exists, get its ID
        parentId = response.files(1).id;
    end
end

% Return the final folder ID
folderId = parentId;
end

function access_token = loadGoogleDrive(pathToToken)

% Load the client credentials
credentials = jsondecode(fileread(pathToToken));

% Define the OAuth 2.0 endpoint
auth_url = 'https://accounts.google.com/o/oauth2/v2/auth';
token_url = 'https://oauth2.googleapis.com/token';

% Define the scopes and redirect URI
scopes = 'https://www.googleapis.com/auth/drive.file';
redirect_uri = 'urn:ietf:wg:oauth:2.0:oob';

% Step 1: Get authorization code
auth_request_url = sprintf('%s?client_id=%s&redirect_uri=%s&response_type=code&scope=%s', ...
    auth_url, credentials.installed.client_id, redirect_uri, scopes);

% Open the authorization URL in the web browser
web(auth_request_url, '-browser');

% User will get a code, which should be pasted here
auth_code = input('Enter the authorization code: ', 's');

% Step 2: Exchange authorization code for access token
options = weboptions('RequestMethod', 'post');

token_response = webwrite(token_url, ...
    'code', auth_code, ...
    'client_id', credentials.installed.client_id, ...
    'client_secret', credentials.installed.client_secret, ...
    'redirect_uri', redirect_uri, ...
    'grant_type', 'authorization_code', ...
    options);

% Extract access token
access_token = token_response.access_token;
end

%------------- END OF CODE -------------