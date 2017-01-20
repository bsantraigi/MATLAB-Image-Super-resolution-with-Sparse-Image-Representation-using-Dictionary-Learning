function [ Y_new ] = repatch( muB, r, p, K, T )
%REPATCH2DATA Summary of this function goes here
% We had N = T*(r-p+1)^2 patches for all images combined
% Repatch will double the implicit patch size, p to 2*p and
% will generate the corresponding data matrix
% New size of data matrix would be N_new = T*(r-2*p+1)^2

% In muB each patch i.e. each column is represented by a 
% K dimensional vector
L = r - p + 1;
L_new = r - 2*p + 1;
Y_new = zeros(4*K, T*L_new^2);

for t = 1:T
    for u = 0:(L_new - 1)
        for v = 1:L_new
            j = (t - 1)*L^2 + v + u*L; % Previous patch index
            j_new = (t - 1)*L_new^2 + v + u*L_new; % New patch index
            newPatch = [muB(:, j), muB(:, j + p);...
                muB(:, j + p*L), muB(:, j + p + p*L)];
            Y_new(:, j_new) = newPatch(:);
        end
    end
end

% m = reshape(1:25, 5, 5)';
% m = [m, m^2];
% mp = im2patch(m, 2);
% mpr = repatch(mp, 5, 2, 4, 2);
% mpu = un_repatch(mpr, 5, 2, 4, 2);

end
