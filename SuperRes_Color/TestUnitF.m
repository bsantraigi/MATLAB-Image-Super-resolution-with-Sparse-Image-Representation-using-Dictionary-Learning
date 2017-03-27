function [ out ] = TestUnitF()
%TESTUNITF Summary of this function goes here
%   Detailed explanation goes here

A = gpuArray(ones(10, 10));
    function [out] = inner(i, j)
        out = i
        out = j
        A(:, i) = A(:, i) + i;
    end
rc = 1:10;
r = gpuArray(rc);
arrayfun(@inner, rc, r);
A
end

