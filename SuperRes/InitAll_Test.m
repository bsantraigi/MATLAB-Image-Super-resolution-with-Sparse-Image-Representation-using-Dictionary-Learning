function [ DH, DL, S, B, PI, ...
    post_PI, biasH, biasL, Gamma, c ] = ...
    InitAll( YH, YL, K, Alpha, Beta, PI )
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

DH = zeros(MH, K);
DL = zeros(ML, K);
S = zeros(K, N);
B = zeros(K, N);
PI = zeros(K, 1); % What is the prob. that D[:, k] is used

Gamma = {};

Gamma.dH = gamrnd(Alpha.d, 1/Beta.d);
Gamma.dL = gamrnd(Alpha.d, 1/Beta.d);
Gamma.s = gamrnd(Alpha.s, 1/Beta.s);
Gamma.nH = gamrnd(Alpha.n, 1/Beta.n);
Gamma.nL = gamrnd(Alpha.n, 1/Beta.n);
Gamma.biasH = gamrnd(Alpha.bias, 1/Beta.bias);
Gamma.biasL = gamrnd(Alpha.bias, 1/Beta.bias);

%% Sampling of D
M = MH;
zmu = zeros(M, 1);
Is = eye(M);
gmd = Gamma.dH;
parfor k = 1:K
    DH(:, k) = mvnrnd(zmu, (1/gmd).*Is);
end

M = ML;
zmu = zeros(M, 1);
Is = eye(M);
gmd = Gamma.dL;
parfor k = 1:K
    DL(:, k) = mvnrnd(zmu, (1/gmd).*Is);
end
clear M
%% Sampling of S
zmu = zeros(K, 1);
Is = eye(K);
gms = Gamma.s;
parfor i = 1:N
    S(:, i) = mvnrnd(zmu, (1/gms).*Is);
end

%% Sampling of bias
M = MH;
biasH = mvnrnd(zeros(M,1), (1/Gamma.biasH)*eye(M))';

M = ML;
biasL = mvnrnd(zeros(M,1), (1/Gamma.biasL)*eye(M))';

clear M
%% Sampling of PI and B (or Z)
alpha_pi = Alpha.pi;
beta_pi = Beta.pi;
parfor k = 1:K %% VERIFIED - parfor is BETTER
    PI(k) = betarnd(alpha_pi, beta_pi);
    for i = 1:N
        B(k, i) = binornd(1, PI(k));
    end
end

%% Sampling of post_PI
post_PI = repmat(PI, 1, N);

fprintf('Initial Samples Ready...\n')

end

