function [ Y ] = sigmoid_Inv( post_PI, restructure )
%SIGMOID_INV Summary of this function goes here
%   Detailed explanation goes here
K = size(post_PI, 1);
N = size(post_PI, 2);
g_PI = gpuArray(post_PI(:));
% g_PI(g_PI <= 1e-5) = 1e-5;
% g_PI(g_PI >= (1 - 1e-5)) = 1 - 1e-5;
    function [v] = sigm_single(v)
        p = v > (1 - 1e-5);
        q = (v < 1e-5);
        r = (v > 1e-5 && v < (1 - 1e-5));
        v = 1e-5*q + (1-1e-5)*p + r*v;
        v = log(v/(1-v));        
    end
% Y = gather(log(g_PI./(1-g_PI)));

v = arrayfun(@sigm_single, g_PI);
Y = gather(reshape(v, K, N));

end

