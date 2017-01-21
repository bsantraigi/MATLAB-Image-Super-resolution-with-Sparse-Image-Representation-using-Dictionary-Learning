function [ out ] = TestUnitF()
%TESTUNITF Summary of this function goes here
%   Detailed explanation goes here

A = ones(10, 10);
    function [out] = inner(i)
        disp(i)
        A(:, i) = A(:, i) + i;
    end
r = 1:10;
arrayfun(@inner, r);
A;
end

