function [ Gamma ] = sampleGammas( Y, D, S, B, bias, Gamma, Alpha, Beta, c )
%SAMPLEGAMMAS Summary of this function goes here
%   Detailed explanation goes here

M = c.M;
N = c.N;
K = c.K;

% Gamma d
Alp = K*M/2 + Alpha.d;
Bet = 0.5*norm(D, 'fro').^2 + Beta.d;
Gamma.d = gamrnd(Alp, 1/Bet);
% Gamma s
Alp = K*N/2 + Alpha.s;
Bet = Beta.s + 0.5 * norm(S, 'fro').^2;
Gamma.s = gamrnd(Alp, 1/Bet);
% Gamma n
Alp = Alpha.n + M*N/2;
Bet = Beta.n + 0.5*norm(Y - D*(S.*B) - repmat(bias, 1, N),'fro').^2;
Gamma.n = gamrnd(Alp, 1/Bet);

% Gamma.bias
Alp = Alpha.bias + M/2;
Bet = Beta.n + 0.5*(bias'*bias);
Gamma.bias = gamrnd(Alp, Bet);
end

