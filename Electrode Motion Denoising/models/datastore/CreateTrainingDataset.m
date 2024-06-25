function CreateTrainingDataset()
% Script CreateTrainingDataset.m will generate the dataset used for
% training Machine / Deep Learning algorithms. It will create a dataset
% with clean reference signals, as well as a range of signals corrupt with
% various amounts of electrode motion noise. The idea is that the reference
% signals will act as a ground truth, and algorithms will learn what
% electrode motion looks like (Both temporal and frequency characteristics).
%
% Step 1: ExtractAndPreProcessNoiseSignal - Extracts the Electrode Motion
% noise file from Physionet. This noise file has been collected through
% specialised placement of electrodes.
%
% Step 2: createSyntheticCleanEcgSignals - Generates synthetic ECG signals
% based on 1st order differential equations. Sampling is used to generate
% signals with a range of morphologies.
%
% Step 3: generateNoisyDatabase - Corrupts the clean ECG signals with
% various amounts of electrode motion noise. 

%% Main processing

% Call function to generate reference noisy signals.
extractAndPreProcessNoiseSignal(); % These get saved in datastore/noiseSignal

% Call function to synthetically generate Clean ECG signals.
createSyntheticCleanEcgSignals();

% Define inputs.
noiseSignalPath = fullfile(erase(mfilename('fullpath'), mfilename), "noiseSignal");
ecgSignalPath = fullfile(erase(mfilename('fullpath'), mfilename), "cleanSignals"); % Create a database of clean ECG // Ensure 500Hz. Keep in .mat format.

% Define input parameters.
maxNosieSections = 10; % Can vary this.
SNR = [0 6 12 18 24];
numberOfGeneratedNoisySignals = 10; % Can vary this.

% Set Google Drive folder IDs (replace these with your actual folder IDs)
googleDriveFolderIDClean = createGoogleDriveFolder('cleanSignals', '');
googleDriveFolderIDNoisy = createGoogleDriveFolder('noisySignals', '');

% Create subdirectories for SNR levels
snrFolderIDs = struct();

for i = 1 : length(SNR)
    folderName = ['SNR', num2str(SNR(i))];
    snrFolderIDs.(folderName) = createGoogleDriveFolder(folderName, googleDriveFolderIDNoisy);
end

% Generate noisy database and upload to Google Drive
generatingNoisyEcgDatabase(noiseSignalPath, ecgSignalPath, 500, maxNosieSections, SNR, numberOfGeneratedNoisySignals, googleDriveFolderIDClean, snrFolderIDs);

end

function folderID = createGoogleDriveFolder(folderName, parentFolderID)
% createGoogleDriveFolder - Creates a folder in Google Drive.
%
% Syntax: folderID = createGoogleDriveFolder(folderName, parentFolderID)
%
% Inputs:
%    folderName - Name of the folder to create.
%    parentFolderID - ID of the parent folder in Google Drive (optional).
%
% Outputs:
%    folderID - ID of the created folder.

%------------- BEGIN CODE --------------
    % Read OAuth 2.0 credentials from JSON file
    credentialsFile = '/Users/benrussell/Frameworks/Google-Drive/client_secret_327364716932-78oacfgib4ilrotdphikdpbkvfnlk76c.apps.googleusercontent.com.json'; % Update this path

    if ~isfile(credentialsFile)
        error('Credentials file not found: %s', credentialsFile);
    end
    
    fid = fopen(credentialsFile, 'r');
    if fid == -1
        error('Cannot open credentials file: %s', credentialsFile);
    end
    
    raw = fread(fid, inf);
    str = char(raw');
    fclose(fid);
    
    credentials = jsondecode(str);

    % Authenticate and get access token
    refreshToken = getRefreshToken();
    accessToken = getAccessTokenUsingRefreshToken(credentials, refreshToken);

    % Define the metadata for the folder
    metadata = struct( ...
        'name', folderName, ...
        'mimeType', 'application/vnd.google-apps.folder');

    if ~isempty(parentFolderID)
        metadata.parents = {parentFolderID};
    end

    % Convert metadata to JSON format
    metadataJson = jsonencode(metadata);

    % Define HTTP header with authorization and content type
    options = weboptions('HeaderFields', { ...
        'Authorization', ['Bearer ', accessToken]; ...
        'Content-Type', 'application/json'}, ...
        'MediaType', 'application/json');


    % Define Google Drive API endpoint for creating a folder
    apiEndpoint = 'https://www.googleapis.com/drive/v3/files';

    % Create the folder
    response = webwrite(apiEndpoint, metadataJson, options);

    % Extract folder ID from response
    folderID = response.id;

    % Display the created folder ID
    fprintf('Created folder ID: %s\n', folderID);

end

function accessToken = getAccessTokenUsingRefreshToken(credentials, refreshToken)
    % getAccessTokenUsingRefreshToken - Get access token using refresh token
    %
    % Syntax: accessToken = getAccessTokenUsingRefreshToken(credentials, refreshToken)
    %
    % Inputs:
    %    credentials - OAuth 2.0 credentials loaded from JSON file.
    %    refreshToken - Refresh token obtained from manual authorization.
    %
    % Outputs:
    %    accessToken - Access token for Google Drive API.

    % Define OAuth 2.0 token endpoint and client details
    tokenEndpoint = 'https://oauth2.googleapis.com/token';
    clientID = credentials.installed.client_id;
    clientSecret = credentials.installed.client_secret;

    % Exchange refresh token for access token
    data = struct('client_id', clientID, 'client_secret', clientSecret, ...
                  'refresh_token', refreshToken, 'grant_type', 'refresh_token');

    % Perform the web request using matlab.net.http package
    import matlab.net.http.*
    import matlab.net.http.field.*
    import matlab.net.http.io.*

    % Convert struct to application/x-www-form-urlencoded format
    fields = fieldnames(data);
    bodyContent = [];
    for i = 1:numel(fields)
        bodyContent = [bodyContent, fields{i}, '=', urlencode(data.(fields{i})), '&'];
    end
    bodyContent(end) = []; % Remove trailing ampersand

    header = [ContentTypeField('application/x-www-form-urlencoded')];
    requestBody = FormProvider(bodyContent);

    request = RequestMessage('POST', header, requestBody);
    response = send(request, matlab.net.URI(tokenEndpoint));

    if response.StatusCode == 200
        accessToken = response.Body.Data.access_token;
    else
        error('Failed to get the access token: %s', char(response.Body.Data));
    end
end



