function [ PI ] = samplePI( B, PI, Alpha, Beta, c )
%SAMPLEPI Summary of this function goes here
%   Detailed explanation goes here

M = c.M;
N = c.N;
K = c.K;
h_PI = gather(PI);
h_B = gather(B);
for k = 1:K
    h_PI(k) = betarnd(Alpha.pi + sum(h_B(k, :)),...
        Beta.pi + N - sum(h_B(k, :)));
end
PI = gpuArray(h_PI);
% for i = 1:N
%     PI(i) = betarnd(Alpha.pi + sum(B(:, i)), Beta.pi + K - sum(B(:, i)));
% end

end

