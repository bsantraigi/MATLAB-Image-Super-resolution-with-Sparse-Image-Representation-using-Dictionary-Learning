%% Load Dictionaries
load('..\WSs_10thSem\8_n_16_coupled_v4_mean_subbed_4or9.mat')
%% Start pll pool
gcp
%% Set folder
imgPath = '../'
typeofimage = 'super_res_test/'

%% Matrix created here
close all
reduceTo_lres = 64;
reduceTo_hres = 128;
patchsize_lres = 4;
patchsize_hres = 9;
column = 1;
totalImages = 2;
overlap_low = 1;
overlap_high = 3;
[YL, means_of_YL] = GetDataMatrix([imgPath 'lres/'],...
    reduceTo_lres, patchsize_lres, totalImages, overlap_low);
[YH, means_of_YH] = GetDataMatrix([imgPath 'hres/'],...
    reduceTo_hres, patchsize_hres, totalImages, overlap_high);
% Y = [YL; YH];
% means_of_Y = [means_of_YL; means_of_YH];
close all
%% Initialize
tic
[X, S, B, c] = InitAll(YH, YL, K1, PI, Gamma);
toc

%% Create 'Lf' - Blurring then Downsampling
pH = patchsize_hres;
pL = patchsize_lres;
pHsQ = pH^2;
pLsQ = pL^2;

LfB = zeros(pHsQ, pHsQ);

sigma_blur = 1;
es = exp(-1/(2*sigma_blur^2));
for r = 1:pHsQ
    % Each row will have 9 non-zero coefficients
    % for the 8 neighbours and its own pixel location
    if floor((r-1)/pH) == 0 || mod(r, pH) == 0 || ...
            mod(r-1, pH) == 0 || ceil(r/pH) == pH
        continue
    end
    LfB(r, r) = 1;
    LfB(r, [r - 1, r + 1, r - pH, r + pH]) = es;
    LfB(r, [r - pH + 1, r - pH - 1, ...
        r + pH + 1, r + pH - 1]) = es^2;
    LfB(r, :) = LfB(r, :)/sum(LfB(r, :));
end

LfD = zeros(pLsQ, pHsQ);

for r = 1:pLsQ
    rx = 2*(floor((r - 1)/pL) + 1);
    ry = 2*(mod(r - 1, pL) + 1);
    rH = (rx - 1)*pH + ry;
    LfD(r, rH) = 1;
end
Lf = LfD * LfB;
%% Covariance matrix of YL - SIGMA
YL_Covar = zeros(pLsQ, pLsQ);
d0 = 0.01;
for r = 1:pLsQ
    for col = 1:pLsQ
        rx = 2*(floor((r - 1)/pL) + 1);
        ry = 2*(mod(r - 1, pL) + 1);
        
        cx = 2*(floor((col - 1)/pL) + 1);
        cy = 2*(mod(col - 1, pL) + 1);
        
        d = (rx - cx)^2 + (ry - cy)^2;
        
        YL_Covar(r, col) = exp( - d/d0^2);
    end
end
figure(2)
clf
imagesc(YL_Covar), colorbar
%% Compare 'Blurred &' vs. 'Just downsampled' images
figure(1)
subplot(2, 2, 1)
imshow(patch2im(YH, 1, 128, 128, 9, means_of_YL, 3));
subplot(2, 2, 2)
imshow(patch2im(Lf * YH, 1, 64, 64, 4, means_of_YL, 1));
subplot(2, 2, 3)
imshow(patch2im(YL, 1, 64, 64, 4, means_of_YL, 1));
subplot(2,2,4)
% M2 = inv(Lf'*(YL_Covar\Lf) + Gamma.nH*eye(c.MH)) *...
%     Lf'*(YL_Covar\YL);
M2 = Lf'*(YL_Covar\YL);
imshow(patch2im(M2, 1, 128, 128, 9, means_of_YL, 3));

%%
% DNA = inv(Lf'*(YL_Covar\Lf) + Gamma.nH*eye(81));

figure(1)
for gr = 1:100
    subplot(1, 3, 1)
    imshow(patch2im(X, 1, 128, 128, 9, means_of_YH, 3));
    drawnow
    tic
    X = NextX(YL, DH, S, B, biasH, Lf, YL_Covar, Gamma, c);
    S = NextS(X, DH, S, B, biasH, Gamma, c);
    [B, ~] = NextB(X, DH, S, B, PI, biasH, Gamma, c);
    toc
end
%%
