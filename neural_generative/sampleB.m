function [ B, post_PI ] = sampleB( Y, D, S, B, PI, post_PI, bias, Gamma, c )
%SAMPLEB Summary of this function goes here
%   Detailed explanation goes here

M = c.M;
N = c.N;
K = c.K;

for k = 1:K
    SB = S.*B;    
%     dtDelY = D(:, k)'*...
%         (Y - repmat(bias, 1, N) - D(:, [1:(k - 1),(k + 1):K])*SB([1:(k - 1),(k + 1):K], :));
    dtDelY = D(:, k)'*...
        (Y - repmat(bias, 1, N) - D(:, [1:(k - 1),(k + 1):K])*SB([1:(k - 1),(k + 1):K], :));
    dTd_k = (D(:, k)'*D(:, k));
    pi_k = PI(k);
    p0 = (1 - pi_k);
    gam_n = Gamma.n;
    
    parfor i = 1:N
        B(k, i) = 1;
%         delY_i = Y(:, i) - bias - D(:, [1:(k - 1),(k + 1):K])*SB([1:(k - 1),(k + 1):K], i);
        arg = S(k, i).^2.*dTd_k - 2*S(k, i)*dtDelY(i);
        p1 = pi_k * exp(-gam_n*arg/2);
        B(k, i) = 0;
        
        if isinf(p1)
            pp = 1;
            B(k, i) = 1;
        else
            s = p1 + p0;
            pp = p1/s;
            B(k, i) = binornd(1, pp);
%             if isnan(B(k,i))
%                 fprintf('NaN Ocurred %d, %d: p0: %f, p1: %f, arg: %f\n', k, i, p0, p1, arg)
%             end
        end
        
        post_PI(k, i) = pp;        
    end
end

end

