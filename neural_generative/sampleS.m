function [ S ] = sampleS(Y, D, S, B, bias, Gamma, c )
%SAMPLES Summary of this function goes here
%   Detailed explanation goes here

M = c.M;
N = c.N;
K = c.K;
Ik = eye(K);
for i = 1:N
    DB = D.*repmat(B(:,i)', M, 1);
    C = inv(Gamma.n.*(DB'*DB) + Gamma.s.*Ik);
    musi = C*(Gamma.n.*DB'*(Y(:, i) - bias));
    S(:, i) = mvnrnd(musi, C);
end

end

