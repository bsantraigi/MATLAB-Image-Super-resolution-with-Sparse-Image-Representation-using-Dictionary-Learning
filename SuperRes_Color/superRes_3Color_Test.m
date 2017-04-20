%% (Dictionary) Coefficient learning follows
%% Load Layer 1
gcp
tic
% load('D:\Users\Bishal Santra\Documents\MATLAB\MTP\SuperRes\WSs_10thSem\8_n_16_coupled_v3_mean_subbed.mat')
toc
%% Create Lf - Blurring then Downsampling
Lf = GetLf(patchsize_hres, patchsize_lres);

%% Set folder
close all
imgPath = './'
typeofimage = 'super_res_test/'

%% Matrix created here
close all
% scale_up = reduceTo_hres/reduceTo_lres;
reduceTo_lres = 85;
reduceTo_hres = reduceTo_lres*scale_up
% patchsize_lres = 4;
% patchsize_hres = 8;
%% Collect Images
column = 1;
imrange = [2, 3, 5];
totalImages = length(imrange);
% overlap_low = 1;
% overlap_high = 2;
[YL2, Cb_of_YL, Cr_of_YL] = GetDataMatrix([imgPath 'lresTest/'],...
    reduceTo_lres, patchsize_lres, imrange, overlap_low);
[YH, Cb_of_YH_original, Cr_of_YH_original] = GetDataMatrix([imgPath 'hresTest/'],...
    reduceTo_hres, patchsize_hres, imrange, overlap_high);

YL = Lf*YH;
Y = [YL];

%%
Cb_of_YH = Cb_of_YH_original*0;
Cr_of_YH = Cr_of_YH_original*0;
for imx = 1:totalImages
    Cb_of_YH(:,:,imx) = imresize(Cb_of_YL(:,:,imx), scale_up, 'bicubic');
    Cr_of_YH(:,:,imx) = imresize(Cr_of_YL(:,:,imx), scale_up, 'bicubic');
end

%% Compare 'Blurred &' vs. 'Just downsampled' images
figure(1)
r = 4;
subplot(1, 3, 1)
imshow(ycbcr2rgb(...
    patch2im(YH, r, reduceTo_hres, reduceTo_hres,...
    patchsize_hres, Cb_of_YH_original, Cr_of_YH_original, overlap_high))...
    );
subplot(1, 3, 2)
imshow(ycbcr2rgb(...
    patch2im(YL, r, reduceTo_lres, reduceTo_lres,...
    patchsize_lres, Cb_of_YL, Cr_of_YL, overlap_low))...
    );
subplot(1, 3, 3)
imshow(ycbcr2rgb(...
    patch2im(YL2, r, reduceTo_lres, reduceTo_lres,...
    patchsize_lres, Cb_of_YL, Cr_of_YL, overlap_low))...
    );
%% Initialize Layer 1

% K1 = 200;
% 
% Alpha1 = {};
% Beta1 = {};
% 
% % Params for gamma distro - LAYER 1
% Alpha1.d = 1e-1;
% Beta1.d = 1e-1;
% Alpha1.s = 4;
% Beta1.s = 4;
% Alpha1.bias = 1e-1;
% Beta1.bias = 1e-1;
% Alpha1.n = 1e-3;
% Beta1.n = 1e-3;
% 
% % Params for beta distro : Near to zero, sparse
% Alpha1.pi = 1;
% Beta1.pi = 1400;

tic
% [ ~, S, B, PI, post_PI, ~, Gamma, c ] = InitAll( Y, K1, Alpha1, Beta1 );
[ S, B, post_PI, c ] =...
    InitAll_Test( YH, YL, K1, Alpha1, Beta1, PI, Gamma );
toc

%% Plot images
training_started=false;
figure(1)
clf
r = 3;
subplot(2, 5, 1)
recon = patch2im(...
    YH, ...
    r, reduceTo_hres, reduceTo_hres,...
    patchsize_hres, isAddingMean*means_of_YH, overlap_high);
imshow(recon);
title('Actual_HRes')       

subplot(2, 5, 4)
recon = patch2im(...
    YL, ...
    r, reduceTo_lres, reduceTo_lres,...
    patchsize_lres, isAddingMean*means_of_YL, overlap_low);
imshow(recon)
title('Recon_LRes')

%% Plot recon
figure(2)
r = 4;
recon = patch2im(...
        Y, ...
        r, reduceTo_lres, reduceTo_lres,...
        patchsize_lres, Cb_of_YL, Cr_of_YL, overlap_low);
sres_bcubic = imresize(recon, 2, 'bicubic');
if n_samples > 0
    S_avg = mean(lastSSamples(:, :, 1:min(last_N, n_samples)), 3);
    B_avg = mean(lastBSamples(:, :, 1:min(last_N, n_samples)), 3);
    B_avg(B_avg > 0.5) = 1;
    B_avg(B_avg < 1) = 0;
else
    S_avg = S;
    B_avg = B;
end
YH_approx = DH*(S.*B) + repmat(biasH, 1, c.N);
YL_approx = DL*(S.*B) + repmat(biasL, 1, c.N);

if n_samples > 0
    YH_m = mean(lastYHSamples(:, :, 1:min(last_N, n_samples)), 3);
else
    YH_m = YH_approx;
end

% Original High Resolution Image
subplot(2, 5, 1)
recon = patch2im(...
    YH, ...
    r, reduceTo_hres, reduceTo_hres,...
    patchsize_hres, Cb_of_YH_original, Cr_of_YH_original, overlap_high);
imshow(ycbcr2rgb(recon));
title('Actual_HRes')

% Reconstructed HIgh resolution image
subplot(2, 5, 2)
reconH = patch2im(...
    YH_m, ...
    r, reduceTo_hres, reduceTo_hres,...
    patchsize_hres, Cb_of_YH, Cr_of_YH, overlap_high);
imshow(ycbcr2rgb(reconH));
title('Recon_HRes')

% Model evaluations
fprintf('PSNR[Bicubic]: %10.8f\n', psnr(sres_bcubic(:,:,1), recon(:,:,1)));
fprintf('PSNR[Model]: %10.8f\n', psnr(reconH(:,:,1), recon(:,:,1)));
fprintf('RMSE[Bicubic]: %10.8f\n', rmse(sres_bcubic(:,:,1), recon(:,:,1)));
fprintf('RMSE[Model]: %10.8f\n', rmse(reconH(:,:,1), recon(:,:,1)));

subplot(2, 5, 3)
%     imshow(patch2im(Y(17:80,(r+1):(r+l)), patchsize_hres))
recon = patch2im(...
    YL, ...
    r, reduceTo_lres, reduceTo_lres,...
    patchsize_lres, Cb_of_YL, Cr_of_YL, overlap_low);
imshow(ycbcr2rgb(recon));
title('Actual_LRes')        

subplot(2, 5, 4)
recon = patch2im(...
    YL_approx, ...
    r, reduceTo_lres, reduceTo_lres,...
    patchsize_lres, Cb_of_YL, Cr_of_YL, overlap_low);
imshow(ycbcr2rgb(recon));
title('Recon_LRes')

subplot(2, 5, 5)
imagesc(B_avg)
title('B Matrix')

subplot(2, 5, 6)
imshow(ycbcr2rgb(sres_bcubic));
title('Bicubic')
%% Sample Stores
n_samples = 0;
last_N = 500;
lastSSamples = zeros(c.K, c.N, last_N);
lastBSamples = zeros(c.K, c.N, last_N);
lastYHSamples = zeros(c.MH, c.N, last_N);
%% Gibbs
normalize = @(Mat) (Mat - min(Mat(:)))/(max(Mat(:)) - min(Mat(:)));
rmse = @(A, ref) sqrt(mean((255*A(:) - 255*ref(:)).^2));
% Pack_n_Show_dictionary(DH, DL);
figure(2)
clf

lastPsnr = -inf;

r = 3;
recon = patch2im(...
        Y, ...
        r, reduceTo_lres, reduceTo_lres,...
        patchsize_lres, Cb_of_YL, Cr_of_YL, overlap_low);
sres_bcubic = imresize(recon, scale_up, 'bicubic');

training_started=true;

tune_length = 10;
round_1 = tune_length;
round_2 = round_1 + tune_length;

mse_array = zeros(2000, 1);
for gr = 1:2000
    % Test here only
    % Calculate averages
    if n_samples > 0
        S_avg = mean(lastSSamples(:, :, 1:min(last_N, n_samples)), 3);
        B_avg = mean(lastBSamples(:, :, 1:min(last_N, n_samples)), 3);
        B_avg(B_avg > 0.5) = 1;
        B_avg(B_avg < 1) = 0;
    else
        S_avg = S;
        B_avg = B;
    end
    YH_approx = DH*(S.*B) + repmat(biasH, 1, c.N);
    YL_approx = DL*(S.*B) + repmat(biasL, 1, c.N);
    
    if n_samples > 0
        YH_m = median(lastYHSamples(:, :, 1:min(last_N, n_samples)), 3);
    else
        YH_m = YH_approx;
    end
%     YH_m = YH_approx;
    
    erH = (sum((YH_approx(:) - YH(:)).^2))/(c.N*c.MH);
    erL = (sum((YL_approx(:) - YL(:)).^2))/(c.N*c.ML);
    fprintf('-----------\n');
    fprintf('MSE: %10.8f, %10.8f\n', erH, erL);
    mse_array(gr) = erH;
    
    % Original High Resolution Image
    subplot(2, 5, 1)
    recon = patch2im(...
        YH, ...
        r, reduceTo_hres, reduceTo_hres,...
        patchsize_hres, Cb_of_YH_original, Cr_of_YH_original, overlap_high);
    imshow(ycbcr2rgb(recon));
    title('Actual_HRes')
    

    % Reconstructed HIgh resolution image
    subplot(2, 5, 2)
    [reconH, reconH2] = patch2im(...
        YH_m, ...
        r, reduceTo_hres, reduceTo_hres,...
        patchsize_hres, Cb_of_YH, Cr_of_YH, overlap_high);
    imshow(ycbcr2rgb(reconH));
    title('Recon_HRes')

    subplot(2, 5, 7)
    imshow(ycbcr2rgb(reconH2));
    subplot(2, 5, 8)
    reconH = (reconH + reconH2)/2;
    imshow(ycbcr2rgb((reconH + reconH2)/2));
    
    % Model evaluations
    thispsnr = psnr(reconH(:,:,1), recon(:,:,1));
    if thispsnr > lastPsnr
        lastPsnr = thispsnr;
    else
        load gong
        sound(y/4, Fs)
        fprintf('PSNR dropped -- BREAKING');
        break
    end
    fprintf('PSNR[Bicubic]: %10.8f\n', psnr(sres_bcubic(:,:,1), recon(:,:,1)));
    fprintf('PSNR[Model]: %10.8f\n', thispsnr);
    fprintf('RMSE[Bicubic]: %10.8f\n', rmse(sres_bcubic(:,:,1), recon(:,:,1)));
    fprintf('RMSE[Model]: %10.8f\n', rmse(reconH(:,:,1), recon(:,:,1)));

    subplot(2, 5, 3)
%     imshow(patch2im(Y(17:80,(r+1):(r+l)), patchsize_hres))
    recon = patch2im(...
        YL, ...
        r, reduceTo_lres, reduceTo_lres,...
        patchsize_lres, Cb_of_YL, Cr_of_YL, overlap_low);
    imshow(ycbcr2rgb(recon));
    title('Actual_LRes')        

    subplot(2, 5, 4)
    recon = patch2im(...
        YL_approx, ...
        r, reduceTo_lres, reduceTo_lres,...
        patchsize_lres, Cb_of_YL, Cr_of_YL, overlap_low);
    imshow(ycbcr2rgb(recon));
    title('Recon_LRes')
    
    subplot(2, 5, 5)
    imagesc(B_avg)
    title('B Matrix')
    
    subplot(2, 5, 6)
    imshow(ycbcr2rgb(sres_bcubic));
    title('Bicubic')

    drawnow
    
    tic
    % LEarn layer 1
    [ ~, ~, S, B, ~, post_PI, ~, ~, ~ ] = ...
        GibbsLevel( YH, YL, DH, DL, S, B, PI, post_PI,...
        biasH, biasL, Gamma, Alpha1, Beta1, c, true );
    
    lastBSamples(:, :, mod(n_samples, last_N) + 1) = B;
    lastSSamples(:, :, mod(n_samples, last_N) + 1) = S;
    lastYHSamples(:, :, mod(n_samples, last_N) + 1) = YH_approx;
    n_samples  = n_samples + 1;
    
    fprintf('[V1_L1]Iteration Complete: %d[%d] \n', gr, n_samples)
    
    if mod(gr, 2) == 0
        if sum(sum(B == 0)) == c.N*K1
            display('Resetting B1')
            [ S, B, post_PI, c ] = InitAll_Test( YH,...
                YL, K1, Alpha1, Beta1, PI, Gamma );
        end
    end
    
    toc

    fprintf('Noise Vars: %3.4f, %3.4f\n', 1/Gamma.nH, 1/Gamma.nL)

end
fprintf('Gibbs Complete...\n')

%% Check RMSE
r = 4;
save_flag = false;
recon = patch2im(...
    Y, ...
    r, reduceTo_lres, reduceTo_lres,...
    patchsize_lres, Cb_of_YL, Cr_of_YL, overlap_low);
sres_bcubic = imresize(recon, scale_up, 'bicubic');
if n_samples > 0
    S_avg = mean(lastSSamples(:, :, 1:min(last_N, n_samples)), 3);
    B_avg = mean(lastBSamples(:, :, 1:min(last_N, n_samples)), 3);
    B_avg(B_avg > 0.5) = 1;
    B_avg(B_avg < 1) = 0;
else
    S_avg = S;
    B_avg = B;
end
YH_approx = DH*(S.*B) + repmat(biasH, 1, c.N);
YL_approx = DL*(S.*B) + repmat(biasL, 1, c.N);

if n_samples > 0
    YH_m = mean(lastYHSamples(:, :, 1:min(last_N, n_samples)), 3);
else
    YH_m = YH_approx;
end
% YH_m = YH_approx;

erH = (sum((YH_approx(:) - YH(:)).^2))/(c.N*c.MH);
erL = (sum((YL_approx(:) - YL(:)).^2))/(c.N*c.ML);
fprintf('-----------\n');
fprintf('MSE: %10.8f, %10.8f\n', erH, erL);
mse_array(gr) = erH;

% Original High Resolution Image
subplot(2, 5, 1)
recon = patch2im(...
    YH, ...
    r, reduceTo_hres, reduceTo_hres,...
    patchsize_hres, Cb_of_YH_original, Cr_of_YH_original, overlap_high);
imshow(ycbcr2rgb(recon));
title('Actual_HRes')

% Reconstructed HIgh resolution image
subplot(2, 5, 2)
[reconH, reconH2] = patch2im(...
    YH_m, ...
    r, reduceTo_hres, reduceTo_hres,...
    patchsize_hres, Cb_of_YH, Cr_of_YH, overlap_high);
imshow(ycbcr2rgb(reconH));
title('Recon_HRes')

subplot(2, 5, 7)
imshow(ycbcr2rgb(reconH2));
subplot(2, 5, 8)
reconH = (reconH + reconH2)/2;
imshow(ycbcr2rgb((reconH + reconH2)/2));

% Save High res and super-resolution output of model
if save_flag
    rx = imrange(r);
    save_as = sprintf('outputs/%dx_hres_%d.png', scale_up, rx);
    imwrite(ycbcr2rgb(recon), save_as)
    save_as = sprintf('outputs/%dx_recon_%d.png', scale_up, rx);
    imwrite(ycbcr2rgb(reconH), save_as)
end

% Model evaluations
fprintf('PSNR[Bicubic]: %10.8f\n', psnr(sres_bcubic(:,:,1), recon(:,:,1)));
fprintf('PSNR[Model]: %10.8f\n', psnr(reconH(:,:,1), recon(:,:,1)));
fprintf('RMSE[Bicubic]: %10.8f\n', rmse(sres_bcubic(:,:,1), recon(:,:,1)));
fprintf('RMSE[Model]: %10.8f\n', rmse(reconH(:,:,1), recon(:,:,1)));

subplot(2, 5, 3)
%     imshow(patch2im(Y(17:80,(r+1):(r+l)), patchsize_hres))
recon = patch2im(...
    YL, ...
    r, reduceTo_lres, reduceTo_lres,...
    patchsize_lres, Cb_of_YL, Cr_of_YL, overlap_low);
imshow(ycbcr2rgb(recon));
title('Actual_LRes')        

subplot(2, 5, 4)
reconL = patch2im(...
    YL_approx, ...
    r, reduceTo_lres, reduceTo_lres,...
    patchsize_lres, Cb_of_YL, Cr_of_YL, overlap_low);
imshow(ycbcr2rgb(reconL));
title('Recon_LRes')

subplot(2, 5, 5)
imagesc(B_avg)
title('B Matrix')

subplot(2, 5, 6)
imshow(ycbcr2rgb(sres_bcubic));
title('Bicubic')

% Save low res and bicubic
if save_flag
    rx = imrange(r);
    save_as = sprintf('outputs/%dx_lres_%d.png', scale_up, rx);
    imwrite(ycbcr2rgb(recon), save_as)
    save_as = sprintf('outputs/%dx_bicubic_%d.png', scale_up, rx);
    imwrite(ycbcr2rgb(sres_bcubic), save_as)
end