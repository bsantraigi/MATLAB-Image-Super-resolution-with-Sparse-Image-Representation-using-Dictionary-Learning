function [ DH, DL, S, B, PI, post_PI, biasH, biasL, Gamma ] = ...
    GibbsLevel( YH, YL, DH, DL, S, B, PI, post_PI, biasH, biasL,...
    Gamma, Alpha, Beta, c, varargin)
%GIBBSLEVEL Summary of this function goes here
%   Detailed explanation goes here

showTime = false;
%% SamplePI - CHECKED
if showTime
    tic
end
PI = samplePI(B, PI, Alpha, Beta, c);
if showTime
    t = toc;
    fprintf('SamplePI took %f s\n', t);
end
%% SampleB - CHECKED
if showTime
    tic
end
[B, post_PI] = sampleB(YH, YL, DH, DL, S, B, PI, post_PI, biasH, biasL, Gamma, c);
if showTime
    t = toc;
    fprintf('SampleB took %f s\n', t);
end

%% SampleGamma - CHECKED
if showTime
    tic
end
Gamma = sampleGammas(YH, YL, DH, DL, S, B,...
    biasH, biasL, Gamma, Alpha, Beta, c);
if showTime
    t = toc;
    fprintf('SampleGamma took %f s\n', t);
end
%% SampleD - CHECKED
if nargin > 14
    doUpdateD = varargin{1};
else
    doUpdateD = true;
end

if showTime
    tic
end
if doUpdateD
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
S = sampleS(YH, YL, DH, DL, S, B, biasH, biasL, Gamma, c);
if showTime
    t = toc;
    fprintf('SampleS took %f s\n', t);
end

%% SampleBIAS - CHECKED
if showTime
    tic
end
if doUpdateD
    biasH = sampleBias(YH, DH, S, B, Gamma, c, 'H');
    biasL = sampleBias(YL, DL, S, B, Gamma, c, 'L');
end
if showTime
    t = toc;
    fprintf('SampleBias took %f s\n', t);
end

end

