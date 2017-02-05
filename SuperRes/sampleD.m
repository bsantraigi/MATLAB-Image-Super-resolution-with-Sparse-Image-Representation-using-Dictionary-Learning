function [ D ] = sampleD(Y, D, S, B, bias, Gamma, c )
%SAMPLED Summary of this function goes here
%   Detailed explanation goes here

M = c.M;
N = c.N;
K = c.K;
SB = S.*B;

Is = eye(M);

%% GPU Stuff
g_SB = gpuArray(SB);
g_D = gpuArray(D);
g_Y = gpuArray(Y);
g_BiasN = gpuArray(repmat(bias, 1, N));
g_gam_n = gpuArray(Gamma.n);
g_gam_d = gpuArray(Gamma.d);
%% Loopy Loop
for k = 1:K
    
    % CPU Code - No parallelization
%     tic
%     Y_approx = D(:,[1:(k - 1), (k + 1):K])*...
%         (SB([1:(k - 1), (k + 1):K], :));
%     delY = (Y - repmat(bias, 1, N)) - Y_approx;
%     % Posterior mu and precision
%     prk = Gamma.n*sum(SB(k, :).^2) + Gamma.d;
%     muk = (Gamma.n/prk).*(delY * SB(k, :)');
%     D(:, k) = mvnrnd(muk, (1/prk).*Is);
%     toc

    % GPU code
    g_D_notk = g_D(:,[1:(k - 1), (k + 1):K]);
    g_S_notk = g_SB([1:(k - 1), (k + 1):K], :);
    Y_approx = g_D_notk*g_S_notk; % Will be on gpu
    delY = (g_Y - g_BiasN) - Y_approx; % Also on gpu
    
    g_SBk = g_SB(k, :);
    % Posterior mu and precision
    
    prk = g_gam_n*norm(g_SBk)^2 + g_gam_d;
    muk = gather((g_gam_n/prk).*(delY * SB(k, :)'));
    prk = gather(prk);
    
    D(:, k) = mvnrnd(muk, (1/prk).*Is);
    g_D(:, k) = D(:, k);
end

end

