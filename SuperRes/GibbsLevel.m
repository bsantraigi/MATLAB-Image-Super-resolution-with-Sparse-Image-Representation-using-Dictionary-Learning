function [ DH, DL, S, B, PI, post_PI, biasH, biasL, Gamma ] = ...
    GibbsLevel( YH, YL, DH, DL, S, B, PI, post_PI, biasH, biasL,...
    Gamma, Alpha, Beta, c, varargin)
%GIBBSLEVEL Summary of this function goes here
%   Detailed explanation goes here

showTime = false;
if nargin > 14
    TestingMode = varargin{1};
    locate = varargin{2};
else
    TestingMode = false;
end
%% SamplePI - CHECKED
if showTime
    tic
end
if ~TestingMode
    PI = samplePI(B, PI, Alpha, Beta, c);
end
if showTime
    t = toc;
    fprintf('SamplePI took %f s\n', t);
end
%% SampleB - CHECKED
if showTime
    tic
end
if ~TestingMode
    [B, post_PI] = sampleB(YH, YL, DH, DL, S, B, PI, post_PI, biasH, biasL, Gamma, c);
else
    [ ~, ~, YHL ] = ReconstructAll( DH, DL,...
        S, B, biasH, biasL, c, YL, locate, true );
    [B, post_PI] = sampleB_Test(YHL, DH, S, B, PI, post_PI, biasH, Gamma, c);
end
if showTime
    t = toc;
    fprintf('SampleB took %f s\n', t);
end

%% SampleGamma - CHECKED
if showTime
    tic
end
if ~TestingMode
    Gamma = sampleGammas(YH, YL, DH, DL, S, B,...
        biasH, biasL, Gamma, Alpha, Beta, c);
end
if showTime
    t = toc;
    fprintf('SampleGamma took %f s\n', t);
end
%% SampleD - CHECKED
if showTime
    tic
end
if ~TestingMode
    DH = sampleD(YH, DH, S, B, biasH, Gamma, c, 'H');
    DL = sampleD(YL, DL, S, B, biasL, Gamma, c, 'L');
end
if showTime
    t = toc;
    fprintf('SampleD took %f s\n', t);
end

%% SampleS - CHECKED
if showTime
    tic
end
if ~TestingMode
    S = sampleS(YH, YL, DH, DL, S, B, biasH, biasL, Gamma, c);
else
    [ ~, ~, YHL ] = ReconstructAll( DH, DL,...
        S, B, biasH, biasL, c, YL, locate, true );
    S = sampleS_Test(YHL, DH, S, B, biasH, Gamma, c);
end
if showTime
    t = toc;
    fprintf('SampleS took %f s\n', t);
end

%% SampleBIAS - CHECKED
if showTime
    tic
end
if ~TestingMode
    biasH = sampleBias(YH, DH, S, B, Gamma, c, 'H');
    biasL = sampleBias(YL, DL, S, B, Gamma, c, 'L');
end
if showTime
    t = toc;
    fprintf('SampleBias took %f s\n', t);
end

end

