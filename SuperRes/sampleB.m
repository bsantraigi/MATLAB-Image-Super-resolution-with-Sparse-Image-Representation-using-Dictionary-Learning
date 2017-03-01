function [ B, post_PI ] = sampleB( ...
    YH, YL, DH, DL, S, B, PI, post_PI, biasH, biasL, Gamma, c )
%SAMPLEB Summary of this function goes here
%   Detailed explanation goes here

MH = c.MH;
ML = c.ML;
N = c.N;
K = c.K;

for k = 1:K
    SB = S.*B;
    notk = [1:(k - 1),(k + 1):K];
%     dtDelY = D(:, k)'*...
%         (Y - repmat(bias, 1, N) - D(:, [1:(k - 1),(k + 1):K])*SB([1:(k - 1),(k + 1):K], :));
    dtDelYH = DH(:, k)'*...
        (YH - repmat(biasH, 1, N) - DH(:, notk)*SB(notk, :));
    dtDelYL = DL(:, k)'*...
        (YL - repmat(biasL, 1, N) - DL(:, notk)*SB(notk, :));
    dTdH_k = (DH(:, k)'*DH(:, k));
    dTdL_k = (DL(:, k)'*DL(:, k));
    pi_k = PI(k);
    p0 = (1 - pi_k);
    gam_nH = Gamma.nH;
    gam_nL = Gamma.nL;
    
    % On GPU
    dtDelYH_gpu = gpuArray(dtDelYH);
    dtDelYL_gpu = gpuArray(dtDelYL);
    Sk_gpu = gpuArray(S(k, :));
    innerH = 0.5*gam_nH*(dTdH_k*(Sk_gpu.^2) - 2*Sk_gpu.*dtDelYH_gpu);
    innerL = 0.5*gam_nL*(dTdL_k*(Sk_gpu.^2) - 2*Sk_gpu.*dtDelYL_gpu);
    
    
    p1_all = gather(pi_k*exp(- innerH - innerL));
    p1_all(p1_all < Inf) =...
        p1_all(p1_all < Inf)./(p1_all(p1_all < Inf) + p0);
    p1_all(isinf(p1_all)) = 1;
    
    post_PI(k, :) = p1_all;
    B(k, :) = binornd(1, p1_all);
    
    % On CPU - parallelized on CPU
%     parfor i = 1:N
%         B(k, i) = 1;
% %         delY_i = Y(:, i) - bias - D(:, [1:(k - 1),(k + 1):K])*SB([1:(k - 1),(k + 1):K], i);
%         arg = S(k, i).^2.*dTd_k - 2*S(k, i)*dtDelY(i);
%         p1 = pi_k * exp(-gam_n*arg/2);
%         B(k, i) = 0;
%         
%         if isinf(p1)
%             pp = 1;
%             B(k, i) = 1;
%         else
%             s = p1 + p0;
%             pp = p1/s;
%             B(k, i) = binornd(1, pp);
% %             if isnan(B(k,i))
% %                 fprintf('NaN Ocurred %d, %d: p0: %f, p1: %f, arg: %f\n', k, i, p0, p1, arg)
% %             end
%         end
%         
%         post_PI(k, i) = pp;        
%     end
end

end

