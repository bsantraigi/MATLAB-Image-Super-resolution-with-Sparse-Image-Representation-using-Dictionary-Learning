function [ Y ] = matnormrnd( M, U, V )
%MATNORMRND Samples Matrix from Matrix Normal Distribution
%   Detailed explanation goes here

A = chol(U)';
B = chol(V);

X = normrnd(0, 1, size(M, 1), size(M,2));

Y = M + A*X*B;

end

