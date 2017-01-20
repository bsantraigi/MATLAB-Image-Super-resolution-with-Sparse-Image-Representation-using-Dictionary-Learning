function [ D, S, B, PI, post_PI, bias, Gamma ] = GibbsLevel( Y, D, S, B, PI, post_PI, bias, Gamma, Alpha, Beta, c, varargin )
%GIBBSLEVEL Summary of this function goes here
%   Detailed explanation goes here

% tic
PI = samplePI(B, PI, Alpha, Beta, c);
% t = toc;
% fprintf('SamplePI took %f s\n', t);

% tic
[B, post_PI] = sampleB(Y, D, S, B, PI, post_PI, bias, Gamma, c);
% t = toc;
% fprintf('SamplePI took %f s\n', t);
% Gamma = sampleGammas(Y, D, S, B, bias, Gamma, Alpha, Beta, c);
% 
% if nargin > 11
%     doUpdateD = varargin{1};
% else
%     doUpdateD = true;
% end
% 
% if doUpdateD
%     D = sampleD(Y, D, S, B, bias, Gamma, c);
% end
% 
% S = sampleS(Y, D, S, B, bias, Gamma, c);
% bias = sampleBias(Y, D, S, B, Gamma, c);

end

