function [ D, S, B, PI, post_PI, bias, Gamma ] = GibbsLevel( Y, D, S, B, PI, post_PI, bias, Gamma, Alpha, Beta, c, varargin )
%GIBBSLEVEL Summary of this function goes here
%   Detailed explanation goes here

showTime = true;
%% SamplePI
if showTime
    tic
end
PI = samplePI(B, PI, Alpha, Beta, c);
if showTime
    t = toc;
    fprintf('SamplePI took %f s\n', t);
end
%% SampleB
if showTime
    tic
end
[B, post_PI] = sampleB(Y, D, S, B, PI, post_PI, bias, Gamma, c);
if showTime
    t = toc;
    fprintf('SampleB took %f s\n', t);
end

%% SampleGamma
if showTime
    tic
end
Gamma = sampleGammas(Y, D, S, B, bias, Gamma, Alpha, Beta, c);
if showTime
    t = toc;
    fprintf('SampleGamma took %f s\n', t);
end
%% SampleD
if nargin > 11
    doUpdateD = varargin{1};
else
    doUpdateD = true;
end

if showTime
    tic
end
if doUpdateD
    D = sampleD(Y, D, S, B, bias, Gamma, c);
end
if showTime
    t = toc;
    fprintf('SampleD took %f s\n', t);
end

%% SampleS
if showTime
    tic
end
S = sampleS(Y, D, S, B, bias, Gamma, c);
if showTime
    t = toc;
    fprintf('SampleS took %f s\n', t);
end

%% SampleBIAS
if showTime
    tic
end
bias = sampleBias(Y, D, S, B, Gamma, c);
if showTime
    t = toc;
    fprintf('SampleBias took %f s\n', t);
end

end

