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

Gamma.d = gamrnd(Alpha.d, 1/Beta.d);
Gamma.s = gamrnd(Alpha.s, 1/Beta.s);
Gamma.n = gamrnd(Alpha.n, 1/Beta.n);
Gamma.bias = gamrnd(Alpha.bias, 1/Beta.bias);

zmu = zeros(M, 1);
Is = eye(M);
for k = 1:K
    D(:, k) = mvnrnd(zmu, (1/Gamma.d).*Is);
end

zmu = zeros(K, 1);
Is = eye(K);
for i = 1:N
    S(:, i) = mvnrnd(zmu, (1/Gamma.s).*Is);
end

bias = mvnrnd(zeros(M,1), (1/Gamma.bias)*eye(M))';

% for i = 1:N
%     PI(i) = betarnd(Alpha.pi, Beta.pi);
% end
for k = 1:K
    PI(k) = betarnd(Alpha.pi, Beta.pi);
    for i = 1:N
        B(k, i) = binornd(1, PI(k));
    end
end

post_PI = repmat(PI, 1, N);

fprintf('Initial Samples Ready...\n')

end

