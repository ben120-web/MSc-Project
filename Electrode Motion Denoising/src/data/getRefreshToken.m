function refreshToken = getRefreshToken()
    % getRefreshToken - Guide the user to get a refresh token manually
    %
    % Syntax: refreshToken = getRefreshToken()
    %
    % Outputs:
    %    refreshToken - Refresh token for Google Drive API.

    % Read OAuth 2.0 credentials from JSON file
    credentialsFile = '/Users/benrussell/Frameworks/Google-Drive/client_secret_327364716932-78oacfgib4ilrotdphikdpbkvfnlk76c.apps.googleusercontent.com.json'; % Update this path

    if ~isfile(credentialsFile)
        error('Credentials file not found: %s', credentialsFile);
    end
    
    fid = fopen(credentialsFile, 'r');
    raw = fread(fid, inf);
    str = char(raw');
    fclose(fid);
    
    credentials = jsondecode(str);

    % Define OAuth 2.0 token endpoint and client details
    clientID = credentials.installed.client_id;
    clientSecret = credentials.installed.client_secret;
    redirectURI = 'urn:ietf:wg:oauth:2.0:oob';  % Updated redirect URI for manual handling

    % Generate authorization URL
    authURL = ['https://accounts.google.com/o/oauth2/auth?response_type=code&client_id=', clientID, ...
               '&redirect_uri=', redirectURI, '&scope=https://www.googleapis.com/auth/drive'];

    % Display the authorization URL and open it in the default web browser
    fprintf('Open the following URL in your browser to authorize the application:\n%s\n', authURL);
    web(authURL, '-browser');

    % Prompt user to enter the authorization code
    authCode = input('Enter the authorization code: ', 's');

    % Exchange authorization code for refresh token
    data = struct('code', authCode, 'client_id', clientID, 'client_secret', clientSecret, ...
                  'redirect_uri', redirectURI, 'grant_type', 'authorization_code');

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
    response = send(request, matlab.net.URI('https://oauth2.googleapis.com/token'));

    if response.StatusCode == 200
        tokenResponse = response.Body.Data;
        refreshToken = tokenResponse.refresh_token;

        % Save the refresh token to a file
        save('refresh_token.mat', 'refreshToken');
    else
        error('Failed to get the refresh token: %s', char(response.Body.Data));
    end
end
