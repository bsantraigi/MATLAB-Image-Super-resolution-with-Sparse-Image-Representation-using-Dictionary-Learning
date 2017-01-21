function [ p_ki_1 ] = SampleB_HelperGPU( pi_k, gam_n, s_ki,...
    dk_norm_sq, dk_delX_i )
%SAMPLEB_HELPERGPU Summary of this function goes here
%   This function only returns P(B_ki = 1)

arg = (s_ki^2)*dk_norm_sq - 2*s_ki*dk_delX_i;
p_ki_1 = pi_k * exp(-gam_n*arg/2);

end

