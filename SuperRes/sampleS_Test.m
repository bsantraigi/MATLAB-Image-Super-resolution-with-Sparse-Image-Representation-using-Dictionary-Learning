function [ S ] = sampleS_Test(Y, D, S, B, bias, Gamma, c )
%SAMPLES Summary of this function goes here
%   Detailed explanation goes here

M = c.ML; % HARD-CODE
N = c.N;
K = c.K;
Ik = eye(K);
gam_n = Gamma.nL; % HARD-CODE
gam_s = Gamma.s;
parfor i = 1:N
    DB = D.*repmat(B(:,i)', M, 1);
    C = inv(gam_n.*(DB'*DB) + gam_s.*Ik);
    musi = C*(gam_n.*DB'*(Y(:, i) - bias));
    S(:, i) = mvnrnd(musi, C);
end

%% Using Arrayfun

% g_D = gpuArray(D);
% g_B = gpuArray(B);
% g_Ik = gpuArray(Ik);
% g_Y_wo_Bias = gpuArray(Y - repmat(bias, 1, N));
% g_S = gpuArray(S);
% 
%     function [out] = elementaryOp(i)
%         %GPU
%         DB = g_D;%.*repmat(g_B(:,i)', M, 1);
%         for kinner = 1:K
%             DB(:, kinner) = DB(:, kinner)*g_B(kinner, i);
%         end
%         C = inv(gam_n.*(DB'*DB) + gam_s.*g_Ik);
%         C = (C + C')/2;
%         musi = C*(gam_n.*DB'*(g_Y_wo_Bias(:, i)));
%         [U_mat, sing_vals, ~] = svd(C);
%         g_S(:, i) = U_mat*sqrt(sing_vals)*randn(K, 1) + musi;
% %         S(:, i) = mvnrnd(gather(musi), gather(C));
%         out = i;
%     end
% tic
% arrayfun(@elementaryOp, gpuArray(1:320));

%% Using gpuArrays

% toc
% for i = 1:N
%     %GPU
%     DB = g_D.*repmat(g_B(:,i)', M, 1);
%     C = inv(gam_n.*(DB'*DB) + gam_s.*g_Ik);
%     C = (C + C')/2;
%     musi = C*(gam_n.*DB'*(g_Y_wo_Bias(:, i)));
%     S(:, i) = mvnrnd(gather(musi), gather(C));
% end



end
