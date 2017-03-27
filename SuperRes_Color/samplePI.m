function [ PI ] = samplePI( B, PI, Alpha, Beta, c )
%SAMPLEPI Summary of this function goes here
%   Detailed explanation goes here

N = c.N;
K = c.K;
for k = 1:K
    PI(k) = betarnd(Alpha.pi + sum(B(k, :)), Beta.pi + N - sum(B(k, :)));
end

% for i = 1:N
%     PI(i) = betarnd(Alpha.pi + sum(B(:, i)), Beta.pi + K - sum(B(:, i)));
% end

end

