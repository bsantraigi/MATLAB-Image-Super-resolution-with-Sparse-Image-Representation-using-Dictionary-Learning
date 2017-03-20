%% (Dictionary) Coefficient learning follows
%% Load Layer 1
gcp
tic
% load('D:\Users\Bishal Santra\Documents\MATLAB\MTP\SuperRes\WSs_10thSem\8_n_16_coupled_v3_mean_subbed.mat')
toc
DL = D(1:patchsize_lres^2, :);
DH = D((1+ patchsize_lres^2):end, :);
DFull = D;
D = DL;

biasFull = bias;
biasL = bias(1:patchsize_lres^2);
biasH = bias((1+patchsize_lres^2):end);
bias = biasL;
%% Create Lf - Blurring then Downsampling
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

%% Set folder
close all
imgPath = './'
typeofimage = 'super_res_test/'

%% Matrix created here
close all
% reduceTo_lres = 64;
% reduceTo_hres = 128;
% patchsize_lres = 4;
% patchsize_hres = 8;
column = 1;
totalImages = 2;
% overlap_low = 1;
% overlap_high = 2;
[YL2, means_of_YL] = GetDataMatrix([imgPath 'lres/'],...
    reduceTo_lres, patchsize_lres, totalImages, overlap_low);
[YH, means_of_YH] = GetDataMatrix([imgPath 'hres/'],...
    reduceTo_hres, patchsize_hres, totalImages, overlap_high);
YL = Lf*YH;
Y = [YL];
% means_of_Y = [means_of_YL; means_of_YH];
%% Compare 'Blurred &' vs. 'Just downsampled' images
figure(1)
subplot(1, 3, 1)
imshow(patch2im(YH, 1, 128, 128, 9, means_of_YH, 3));
subplot(1, 3, 2)
imshow(patch2im(YL, 1, 64, 64, 4, means_of_YH, 1));
subplot(1, 3, 3)
imshow(patch2im(YL2, 1, 64, 64, 4, means_of_YH, 1));
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
% Beta1.pi = 1800;

tic
% [ ~, S, B, PI, post_PI, ~, Gamma, c ] = InitAll( Y, K1, Alpha1, Beta1 );
[ S, B, post_PI, c ] =...
    InitAll_Test( YH, YL, K1, Alpha1, Beta1, PI, Gamma );
toc

%% Plot images
training_started=false;
figure(1)
clf
r = 1;
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
r = 1;
if n_samples > 0
    S_avg = mean(lastSSamples(:, :, 1:min(100, n_samples)), 3);
    B_avg = mean(lastBSamples(:, :, 1:min(100, n_samples)), 3);
%     B_avg(B_avg > 0.95) = 1;
%     B_avg(B_avg < 1) = 0;
else
    S_avg = S;
    B_avg = B;
end
YH_approx = DH*(S_avg.*B_avg) + repmat(biasH, 1, c.N);
YL_approx = DL*(S_avg.*B_avg) + repmat(biasL, 1, c.N);

subplot(2, 5, 1)
recon = patch2im(...
    YH, ...
    r, reduceTo_hres, reduceTo_hres,...
    patchsize_hres, isAddingMean*means_of_YH, overlap_high);
imshow(recon);
title('Actual_HRes')

subplot(2, 5, 2)
recon = patch2im(...
    YH_approx, ...
    r, reduceTo_hres, reduceTo_hres,...
    patchsize_hres, isAddingMean*means_of_YH, overlap_high);
imshow(recon)
title('Recon_HRes')

subplot(2, 5, 3)
%     imshow(patch2im(Y(17:80,(r+1):(r+l)), patchsize_hres))
recon = patch2im(...
    YL, ...
    r, reduceTo_lres, reduceTo_lres,...
    patchsize_lres, isAddingMean*means_of_YL, overlap_low);
imshow(recon);
title('Actual_LRes')        

subplot(2, 5, 4)
recon = patch2im(...
    YL_approx, ...
    r, reduceTo_lres, reduceTo_lres,...
    patchsize_lres, isAddingMean*means_of_YL, overlap_low);
imshow(recon)
title('Recon_LRes')

subplot(2, 5, 5)
imagesc(B)
title('B Matrix')

subplot(2, 5, 6)
recon = patch2im(...
    Y, ...
    r, reduceTo_lres, reduceTo_lres,...
    patchsize_lres, isAddingMean*means_of_YL, overlap_low);
sres_bcubic = imresize(recon, 2, 'bicubic');
imshow(sres_bcubic)
title('Bicubic')
%% Sample Stores
n_samples = 0;
last_N = 50;
lastSSamples = zeros(c.K, c.N, last_N);
lastBSamples = zeros(c.K, c.N, last_N);
lastYHSamples = zeros(c.MH, c.N, last_N);
%% Gibbs
normalize = @(Mat) (Mat - min(Mat(:)))/(max(Mat(:)) - min(Mat(:)));
% Pack_n_Show_dictionary(DH, DL);
figure(2)
clf

r = 1;
isAddingMean = true;
recon = patch2im(...
        Y, ...
        r, reduceTo_lres, reduceTo_lres,...
        patchsize_lres, isAddingMean*means_of_YL, overlap_low);
sres_bcubic = imresize(recon, 2, 'bicubic');

training_started=true;

tune_length = 10;
round_1 = tune_length;
round_2 = round_1 + tune_length;

mse_array = zeros(2000, 1);
for gr = 1:2000
    % Test here only
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
    YL_approx = DL*(S_avg.*B_avg) + repmat(biasL, 1, c.N);
    
    if n_samples > 0
        YH_m = mean(lastYHSamples(:, :, 1:min(last_N, n_samples)), 3);
    else
        YH_m = YH_approx;
    end
    
    erH = (sum((YH_approx(:) - YH(:)).^2))/(c.N*c.MH);
    erL = (sum((YL_approx(:) - YL(:)).^2))/(c.N*c.ML);
    fprintf('-----------\n');
    fprintf('MSE: %10.8f, %10.8f\n', erH, erL);
    mse_array(gr) = erH;
    
    subplot(2, 5, 1)
    recon = patch2im(...
        YH, ...
        r, reduceTo_hres, reduceTo_hres,...
        patchsize_hres, isAddingMean*means_of_YH, overlap_high);
    imshow(recon);
    title('Actual_HRes')
    fprintf('PSNR[Bicubic]: %10.8f\n', psnr(sres_bcubic, recon));

    subplot(2, 5, 2)
    recon = patch2im(...
        YH_m, ...
        r, reduceTo_hres, reduceTo_hres,...
        patchsize_hres, isAddingMean*means_of_YH, overlap_high);
    imshow(recon)
    title('Recon_HRes')

    subplot(2, 5, 3)
%     imshow(patch2im(Y(17:80,(r+1):(r+l)), patchsize_hres))
    recon = patch2im(...
        YL, ...
        r, reduceTo_lres, reduceTo_lres,...
        patchsize_lres, isAddingMean*means_of_YL, overlap_low);
    imshow(recon);
    title('Actual_LRes')        

    subplot(2, 5, 4)
    recon = patch2im(...
        YL_approx, ...
        r, reduceTo_lres, reduceTo_lres,...
        patchsize_lres, isAddingMean*means_of_YL, overlap_low);
    imshow(recon)
    title('Recon_LRes')
    
    subplot(2, 5, 5)
    imagesc(B_avg)
    title('B Matrix')
    
    subplot(2, 5, 6)
    imshow(sres_bcubic)
    title('Bicubic')
    
    subplot(2, 5, 7)
    recon = patch2im(...
        YH, ...
        r, reduceTo_hres, reduceTo_hres,...
        patchsize_hres, 0*means_of_YH, overlap_high);
    imshow(recon);
    title('Actual_HRes')

    subplot(2, 5, 8)
    recon_model = patch2im(...
        YH_m, ...
        r, reduceTo_hres, reduceTo_hres,...
        patchsize_hres, 0*means_of_YH, overlap_high);
    imshow(recon_model)
    title('Recon_HRes')
    fprintf('PSNR[Model]: %10.8f\n', psnr(recon_model, recon));
    
    subplot(2, 5, 9)
%     imshow(patch2im(Y(17:80,(r+1):(r+l)), patchsize_hres))
    recon = patch2im(...
        YL, ...
        r, reduceTo_lres, reduceTo_lres,...
        patchsize_lres, 0*means_of_YL, overlap_low);
    imshow(recon);
    title('Actual_LRes')        

    subplot(2, 5, 10)
    recon = patch2im(...
        YL_approx, ...
        r, reduceTo_lres, reduceTo_lres,...
        patchsize_lres, 0*means_of_YL, overlap_low);
    imshow(recon)
    title('Recon_LRes')
    
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
    
    fprintf('[V1_L1]Iteration Complete: %d \n', gr)
    
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

%% Optimizer - Prepare Code
% Lf matrix - Down-Sampler matrix
locate = (repmat(8*(1:2:8)', 1, 4)+repmat((1:2:8), 4, 1) + 1)';
locate = locate(:);
Lf = eye(64, 64);
Lf = Lf(locate, :);

% Test Lf Matrix
p = 1:64; p = reshape(p, 8, 8); p
imresize(p, 0.5, 'nearest')
L = eye(8, 8);
L = L(2:2:8, :);
R = eye(8, 8);
R = R(:, 2:2:8);
L*p*R;
reshape(Lf * p(:), 4, 4)

%% Optimizer

% Lf matrix - Down-Sampler matrix
% locate = (repmat(...
%     patchsize_hres*(1:2:patchsize_hres)', 1, patchsize_lres)+...
%     repmat((1:2:patchsize_hres), patchsize_lres, 1) + 1)';
% locate = locate(:);
% Lf = eye(patchsize_hres^2);
% Lf = Lf(locate, :);

SNew = S_avg;
% Hyperparameters
k1 = 1; % HyperParam for body similarity
k2 = 0.2; % HyperParam for similarity with low res image

% Define DHU and DHL
DHU = DH(1:(overlap_high*patchsize_hres), :);
DHL = DH((end - overlap_high*patchsize_hres + 1):end, :);
DHBody = DH(...
    (overlap_high*patchsize_hres):(end - overlap_high*patchsize_hres + 1), :);
LfDH = (Lf*DH);
% Calculate delJ
figure(4)
for iter_counter = 1:30
    if mod(iter_counter, 10) == 0
        fprintf('Iteration: %d\n',  iter_counter);
    end
    delJ = 0*S;
    for i = 2:size(S,2)
        delT1 = -2*(DHU'*DHL)*(SNew(:,i - 1).*B(:, i -1)) +...
            2*(DHU'*DHU)*(SNew(:,i).*B(:,i));
        delT2 = -2*k1*(DHBody'*DHBody)*(S_avg(:,i).*B(:,i)) + ...
            2*k1*(DHBody'*DHBody)*(SNew(:,i).*B(:,i));
        delT3 = -2*k2*LfDH'*Y(:, i) + ...
            2*k2*(LfDH'*LfDH)*(SNew(:,i).*B(:,i));
        delJ(:, i) = delT1 + delT2 + delT3;
    end
    
    learning_rate = 0.01; 
    SNew = SNew - learning_rate * delJ;
    
    which_image_2plot = 1;
    isAddingMean = true;
    step = (reduceTo_hres/patchsize_hres)^2;
    subplot(2, 5, 1)
    recon = patch2im(...
        YH, ...
        which_image_2plot, reduceTo_hres, reduceTo_hres,...
        patchsize_hres, isAddingMean*means_of_YH, overlap_high);
    imshow(recon);
    title('Actual_HRes')

    Y_approxH = DH*(S_avg.*B) + repmat(biasH, 1, c.N);
    subplot(2, 5, 2)
    recon = patch2im(...
        Y_approxH, ...
        which_image_2plot, reduceTo_hres, reduceTo_hres,...
        patchsize_hres, isAddingMean*means_of_YL, overlap_high);
    imshow(recon)
    title('Recon_HRes')

    Y_approxH = DH*(SNew.*B) + repmat(biasH, 1, c.N);
    subplot(2, 5, 3)
    recon = patch2im(...
        Y_approxH, ...
        which_image_2plot, reduceTo_hres, reduceTo_hres,...
        patchsize_hres, isAddingMean*means_of_YH, overlap_high);
    imshow(recon)
    title('PostRecon_HRes')
    drawnow
end

%% Just plot post-processed version
figure(5)
which_image_2plot = 1;
isAddingMean = true;
step = (reduceTo_hres/patchsize_hres)^2;
subplot(2, 5, 1)
recon = patch2im(...
    YH, ...
    which_image_2plot, reduceTo_hres, reduceTo_hres,...
    patchsize_hres, isAddingMean*means_of_YH, overlap_high);
imshow(recon);
title('Actual_HRes')

Y_approxH = DH*(S.*B) + repmat(biasH, 1, c.N);
subplot(2, 5, 2)
recon = patch2im(...
    Y_approxH, ...
    which_image_2plot, reduceTo_hres, reduceTo_hres,...
    patchsize_hres, isAddingMean*means_of_YL, overlap_high);
imshow(recon)
title('Recon_HRes')

Y_approxH = DH*(SNew.*B) + repmat(biasH, 1, c.N);
subplot(2, 5, 3)
recon = patch2im(...
    Y_approxH, ...
    which_image_2plot, reduceTo_hres, reduceTo_hres,...
    patchsize_hres, isAddingMean*means_of_YL, overlap_high);
% Sharpen
h = fspecial('sobel');
reconSobelH = imfilter(recon, h);
reconSobelV = imfilter(recon, h');
enhanceFrac = 0.1;
% recon = recon + reconSobelH*enhanceFrac + reconSobelV*enhanceFrac;
% Then plot
imshow(recon)
title('PostRecon_HRes')

subplot(2, 5, 4)
recon = patch2im(...
    Y, ...
    which_image_2plot, reduceTo_lres, reduceTo_lres,...
    patchsize_lres, isAddingMean*means_of_YL, overlap_low);
imshow(recon);
title('Actual_LRes')  

recon = patch2im(...
    Y, ...
    which_image_2plot, reduceTo_lres, reduceTo_lres,...
    patchsize_lres, isAddingMean*means_of_YL, overlap_low);
sres_bcubic = imresize(recon, 2, 'bicubic');
subplot(2, 5, 5)
imshow(sres_bcubic)
title('Bicubic')
drawnow