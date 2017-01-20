function [ S ] = sampleS(Y, D, S, B, bias, Gamma, c )
%SAMPLES Summary of this function goes here
%   Detailed explanation goes here

M = c.M;
N = c.N;
K = c.K;
Ik = eye(K);
gam_n = Gamma.n;
gam_s = Gamma.s;
parfor i = 1:N
    DB = D.*repmat(B(:,i)', M, 1);
    C = inv(gam_n.*(DB'*DB) + gam_s.*Ik);
    musi = C*(gam_n.*DB'*(Y(:, i) - bias));
    S(:, i) = mvnrnd(musi, C);
end

end

