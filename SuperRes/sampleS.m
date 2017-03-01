function [ S ] = sampleS(YH, YL, DH, DL, S, B, biasH, biasL, Gamma, c )
%SAMPLES Summary of this function goes here
%   Detailed explanation goes here

MH = c.MH;
ML = c.ML;
N = c.N;
K = c.K;
Ik = eye(K);
gam_nH = Gamma.nH;
gam_nL = Gamma.nL;
gam_s = Gamma.s;
parfor i = 1:N
    DHB = DH.*repmat(B(:,i)', MH, 1);
    DLB = DL.*repmat(B(:,i)', ML, 1);
    C = inv(gam_nH.*(DHB'*DHB) + gam_nL.*(DLB'*DLB) + gam_s.*Ik);
    musi = C*(gam_nH.*DHB'*(YH(:, i) - biasH) + ...
        gam_nL.*DLB'*(YL(:, i) - biasL));
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

