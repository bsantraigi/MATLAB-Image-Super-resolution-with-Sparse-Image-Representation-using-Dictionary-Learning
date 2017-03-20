function [ X ] = NextX( Y, DH, S, B, biasH, Lf, YL_Covar, Gamma, c )
%NEXTX Summary of this function goes here
%   Detailed explanation goes here

% Lf^T Sigma^-1 Lf + gam_nh*I
biasH = repmat(biasH, 1, c.N);
M1 = inv(Lf'*(YL_Covar\Lf) + Gamma.nH*eye(c.MH));
means_of_X = Lf'*(YL_Covar\Y) + Gamma.nH*(DH*(S.*B) + biasH);
for i = 1:c.N
    X(:, i) = mvnrnd(means_of_X(:, i), M1);
end

end

