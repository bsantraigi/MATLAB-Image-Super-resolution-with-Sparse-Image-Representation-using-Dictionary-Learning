function Pack_n_Show_dictionary( DH, DL )
%PACK_N_SHOW_DICTIONARY Summary of this function goes here
%   Detailed explanation goes here

    function [imMat] = getFImage(D)
        L = sqrt(size(D, 1));
        L2 = L;
        H = ceil(sqrt(size(D, 2)));
        normalize = @(Mat) (Mat - min(Mat(:)))/(max(Mat(:)) - min(Mat(:)));
        imMat = zeros(L*H, L*H);
        for i = 1:size(D, 2)
            x = (mod((i - 1), H))*(L2 + 2) + 1;
            y = (floor((i - 1)/H))*(L2 + 2) + 1;
            imMat(y:(y + L2 - 1), x:(x + L2 - 1)) = normalize(imresize(reshape(D(:, i), L, L), L2/L, 'nearest'));
        end
    end

fim = getFImage(DH);
figure(197)
subplot(1, 2, 1);
imshow(fim);
title('High res dictionary patches')

fim = getFImage(DL);
figure(197)
subplot(1, 2, 2);
imshow(fim);
title('Low res dictionary patches')

end

