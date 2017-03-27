function [ S, B, post_PI, c ] = ...
    InitAll_Test( YH, YL, K, Alpha, Beta, PI, Gamma )
%INITALL Summary of this function goes here
%   Detailed explanation goes here

c = {};
MH = size(YH, 1);
ML = size(YL, 1);
N = size(YH, 2);
c.MH = MH;
c.ML = ML;
c.N = N;
c.K = K;


S = zeros(K, N);
B = zeros(K, N);

%% Sampling of S
zmu = zeros(K, 1);
Is = eye(K);
gms = Gamma.s;
parfor i = 1:N
    S(:, i) = mvnrnd(zmu, (1/gms).*Is);
end

%% Sampling of bias
% M = MH;
% biasH = mvnrnd(zeros(M,1), (1/Gamma.biasH)*eye(M))';
% 
% M = ML;
% biasL = mvnrnd(zeros(M,1), (1/Gamma.biasL)*eye(M))';
% 
% clear M
%% Sampling of PI and B (or Z)
alpha_pi = Alpha.pi;
beta_pi = Beta.pi;
parfor k = 1:K %% VERIFIED - parfor is BETTER
    % PI(k) = betarnd(alpha_pi, beta_pi);
    for i = 1:N
        B(k, i) = binornd(1, PI(k));
    end
end

%% Sampling of post_PI
post_PI = repmat(PI, 1, N);

fprintf('Initial Samples Ready...\n')

end

