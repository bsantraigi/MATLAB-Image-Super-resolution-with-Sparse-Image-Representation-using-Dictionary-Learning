function [ D, S, B, PI, post_PI, bias, Gamma, c ] = InitAll( Y, K, Alpha, Beta )
%INITALL Summary of this function goes here
%   Detailed explanation goes here

c = {};
M = size(Y, 1);
N = size(Y, 2);
c.M = M;
c.N = N;
c.K = K;

D = zeros(M, K);
S = zeros(K, N);
B = zeros(K, N);
PI = zeros(K, 1);


Gamma = {};

Gamma.d = gpuArray(gamrnd(Alpha.d, 1/Beta.d));
Gamma.s = gpuArray(gamrnd(Alpha.s, 1/Beta.s));
Gamma.n = gpuArray(gamrnd(Alpha.n, 1/Beta.n));
Gamma.bias = gpuArray(gamrnd(Alpha.bias, 1/Beta.bias));

zmu = zeros(M, 1);
Is = eye(M);
for k = 1:K
    D(:, k) = mvnrnd(zmu, (1/Gamma.d).*Is);
end
% Convert to gpu array
D = gpuArray(D);


zmu = zeros(K, 1);
Is = eye(K);
for i = 1:N
    S(:, i) = mvnrnd(zmu, (1/Gamma.s).*Is);
end
% Convert to gpu array
S = gpuArray(S);

bias = mvnrnd(zeros(M,1), (1/Gamma.bias)*eye(M))';
% Convert to gpu array
bias = gpuArray(bias);

% for i = 1:N
%     PI(i) = betarnd(Alpha.pi, Beta.pi);
% end
parfor k = 1:K %% VERIFIED - parfor is BETTER
    PI(k) = betarnd(Alpha.pi, Beta.pi);
    for i = 1:N
        B(k, i) = binornd(1, PI(k));
    end
end

post_PI = repmat(PI, 1, N);

fprintf('Initial Samples Ready...\n')

end

