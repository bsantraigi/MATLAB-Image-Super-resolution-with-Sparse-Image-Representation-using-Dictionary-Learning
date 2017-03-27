function [ muD, muS, PI, Gamma, Palpha, Pbeta, c ] = InitAll_VB( Y, K, Alpha, Beta )
%INITALL Summary of this function goes here
%   Detailed explanation goes here
c = {};
M = size(Y, 1);
N = size(Y, 2);
c.M = M;
c.N = N;
c.K = K;

PI = zeros(1, N);

Gamma = {};

Gamma.d = repmat(gamrnd(Alpha.d, 1/Beta.d), K, 1);
Gamma.PrecS = zeros(K,K,N);
for i = 1:N
    f = 1/gamrnd(Alpha.s, 1/Beta.s);
    Gamma.PrecS(:,:,i) = inv(f*eye(K));
    Gamma.CovS(:,:,i) = inv(Gamma.PrecS(:,:,i));
end


Gamma.n = gamrnd(Alpha.n, 1/Beta.n);

for i = 1:N
    PI(i) = betarnd(Alpha.pi/K, Beta.pi*(K-1)/K);
end
PI = repmat(PI, K, 1);

muD = randn(M, K);
muS = randn(K, N);

Palpha = Alpha;
Pbeta = Beta;

Palpha.pi = repmat(Alpha.pi, K, 1);
Pbeta.pi = repmat(Beta.pi, K, 1);

fprintf('VB ready to start...\n')
end

