function [ Y ] = sigmoid_Inv( PI_Mat )
%SIGMOID_INV Summary of this function goes here
%   Detailed explanation goes here
PI_Mat(PI_Mat <= 1e-5) = 0.0001;
PI_Mat(PI_Mat >= (1 - 1e-5)) = 0.9999;
Y = log(PI_Mat./(1-PI_Mat));
end

