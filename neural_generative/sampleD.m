function [ D ] = sampleD(Y, D, S, B, bias, Gamma, c )
%SAMPLED Summary of this function goes here
%   Detailed explanation goes here

M = c.M;
N = c.N;
K = c.K;
SB = S.*B;
Is = eye(M);
for k = 1:K
    Y_approx = D(:,[1:(k - 1), (k + 1):K])*...
        (SB([1:(k - 1), (k + 1):K], :));
    delY = (Y - repmat(bias, 1, N)) - Y_approx;
    % Posterior mu and precision
    prk = Gamma.n*sum(SB(k, :).^2) + Gamma.d;
    muk = (Gamma.n/prk).*(delY * SB(k, :)');
    D(:, k) = mvnrnd(muk, (1/prk).*Is);
end

end

