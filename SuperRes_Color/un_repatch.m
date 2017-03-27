function [ muB ] = un_repatch( Y_new, r, p, K, T )
%UNPATCH Summary of this function goes here
%   Detailed explanation goes here

L = r - p + 1;
L_new = r - 2*p + 1;
muB = zeros(K, T*L^2);

for t = 1:T
    for u = 0:(L_new - 1)
        for v = 1:L_new
            j = v + u*L; % Previous patch index
            j_new = v + u*L_new; % New patch index
            
            j = (t - 1)*L^2 + v + u*L; % Previous patch index
            j_new = (t - 1)*L_new^2 + v + u*L_new; % New patch index
            
            newPatch = Y_new(:, j_new);
            muB(:, j) = newPatch(1:K);
            muB(:, j + p*L) = newPatch((K + 1):2*K);
            muB(:, j + p) = newPatch((2*K + 1):3*K);            
            muB(:, j + p + p*L) = newPatch((3*K + 1):4*K);
%             newPatch = [muB(:, j), muB(:, j + p);...
%                 muB(:, j + p*L), muB(:, j + p + p*L)];
%             Y_new(:, (t - 1)*L_new^2 + j_new) = newPatch(:);
        end
    end
end

end

