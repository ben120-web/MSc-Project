function accessToken = getRefreshToken()
    % Load the OAuth 2.0 credentials
    credentialsFile = '/Users/benrussell/Frameworks/Google-Drive/client_secret_327364716932-78oacfgib4ilrotdphikdpbkvfnlk76c.apps.googleusercontent.com.json'; % Update this path
    if ~isfile(credentialsFile)
        error('Credentials file not found: %s', credentialsFile);
    end
    
    fid = fopen(credentialsFile, 'r');
    raw = fread(fid, inf);
    str = char(raw');
    fclose(fid);
    credentials = jsondecode(str);

    clientID = credentials.installed.client_id;
    clientSecret = credentials.installed.client_secret;
    redirectURI = 'urn:ietf:wg:oauth:2.0:oob';

    % Construct the authorization URL
    authURL = ['https://accounts.google.com/o/oauth2/auth?response_type=code&client_id=', clientID, ...
               '&redirect_uri=', redirectURI, '&scope=https://www.googleapis.com/auth/drive&access_type=offline'];
    fprintf('Opening the following URL in your browser to authorize the application:\n%s\n', authURL);
    
    % Open the authorization URL in the default web browser
    web(authURL, '-browser');
    
    % Prompt the user to enter the authorization code
    authCode = input('Enter the authorization code: ', 's');

    % Exchange authorization code for refresh token
    tokenEndpoint = 'https://oauth2.googleapis.com/token';
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
    response = send(request, matlab.net.URI(tokenEndpoint));

    if response.StatusCode == 200
        refreshToken = response.Body.Data.refresh_token;
        accessToken = response.Body.Data.access_token;

        % Display the refresh token
        fprintf('Refresh Token: %s\n', refreshToken);
        
        % Save the refresh token to a file for future use
        fid = fopen('refresh_token.txt', 'w');
        fprintf(fid, '%s', refreshToken);
        fclose(fid);

        % Save the access token to a file for immediate use
        fprintf('Access Token: %s\n', accessToken);
        fid = fopen('access_token.txt', 'w');
        fprintf(fid, '%s', accessToken);
        fclose(fid);
    else
        error('Failed to get the refresh token: %s', char(response.Body.Data));
    end
end

function encoded = urlencode(str)
    % URL-encode a string
    encoded = char(java.net.URLEncoder.encode(str, 'UTF-8'));
end
