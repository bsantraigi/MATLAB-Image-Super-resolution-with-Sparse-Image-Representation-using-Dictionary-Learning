function [ Gamma ] = sampleGammas( YH, YL, DH, DL, S, B, biasH, biasL, Gamma, Alpha, Beta, c )
%SAMPLEGAMMAS Summary of this function goes here
%   Detailed explanation goes here

MH = c.MH;
ML = c.ML;
N = c.N;
K = c.K;

% Gamma dH
Alp = K*MH/2 + Alpha.d;
Bet = 0.5*norm(DH, 'fro').^2 + Beta.d;
Gamma.dH = gamrnd(Alp, 1/Bet);
% Gamma dL
Alp = K*ML/2 + Alpha.d;
Bet = 0.5*norm(DL, 'fro').^2 + Beta.d;
Gamma.dL = gamrnd(Alp, 1/Bet);

% Gamma s
Alp = K*N/2 + Alpha.s;
Bet = Beta.s + 0.5 * norm(S, 'fro').^2;
Gamma.s = gamrnd(Alp, 1/Bet);

% Gamma nH
Alp = Alpha.n + MH*N/2;
Bet = Beta.n + 0.5*norm(YH - DH*(S.*B) - repmat(biasH, 1, N),'fro').^2;
Gamma.nH = gamrnd(Alp, 1/Bet);
% Gamma nL
Alp = Alpha.n + ML*N/2;
Bet = Beta.n + 0.5*norm(YL - DL*(S.*B) - repmat(biasL, 1, N),'fro').^2;
Gamma.nL = gamrnd(Alp, 1/Bet);

% Gamma.biasH
Alp = Alpha.bias + MH/2;
Bet = Beta.n + 0.5*(biasH'*biasH);
Gamma.biasH = gamrnd(Alp, Bet);
% Gamma.biasH
Alp = Alpha.bias + ML/2;
Bet = Beta.n + 0.5*(biasL'*biasL);
Gamma.biasL = gamrnd(Alp, Bet);

end

