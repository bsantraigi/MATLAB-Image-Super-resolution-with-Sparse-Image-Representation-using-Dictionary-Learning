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
% overlap_low = 2;
% overlap_high = 4;
[YL, means_of_YL] = GetDataMatrix([imgPath 'lres/'],...
    reduceTo_lres, patchsize_lres, totalImages, overlap_low);
[YH, means_of_YH] = GetDataMatrix([imgPath 'hres/'],...
    reduceTo_hres, patchsize_hres, totalImages, overlap_high);
Y = [YL];
means_of_Y = [means_of_YL; means_of_YH];
%% Initialize Layer 1

% K1 = 200;
% 
% Alpha1 = {};
% Beta1 = {};
% 
% % Params for gamma distro - LAYER 1
Alpha1.d = 1e-1;
Beta1.d = 1e-1;
Alpha1.s = 4;
Beta1.s = 4;
Alpha1.bias = 1e-1;
Beta1.bias = 1e-1;
Alpha1.n = 1e-3;
Beta1.n = 1e-3;
% 
% % Params for beta distro : Near to zero, sparse
Alpha1.pi = 1;
Beta1.pi = 1800;

tic
% [ ~, S, B, PI, post_PI, ~, Gamma, c ] = InitAll( Y, K1, Alpha1, Beta1 );
[ ~, ~, S, B, ~, post_PI, ~, ~, ~, ~ ] =...
    InitAll( YH, YL, K1, Alpha1, Beta1 );
toc

%% Plot images
training_started=false;
figure(1)
clf
which_image_2plot = 1;
subplot(2, 5, 1)
imshow(...
    normalize(...
    patch2im(YH(:,(1 + (which_image_2plot-1)*step):(which_image_2plot*step)), patchsize_hres)));
title('Actual_HRes')        

subplot(2, 5, 3)
imshow(...
    normalize(...
    patch2im(...
    YL(:,(1 + (which_image_2plot-1)*step):(which_image_2plot*step)),...
    patchsize_lres))...
    );
title('Actual_LRes') 

%% Plot recon
if training_started
    figure(2)
    clf
    which_image_2plot = 1;
    subplot(2, 5, 1)
%     imshow(patch2im(Y(17:80,(r+1):(r+l)), patchsize_hres))
    imshow(...
        normalize(...
        patch2im(YH(:,(1 + (which_image_2plot-1)*step):(which_image_2plot*step)), patchsize_hres))...        
    );
    title('Actual_HRes')

    subplot(2, 5, 2)
    recon = patch2im(Y_approxH(:,(1 + (which_image_2plot-1)*step):(which_image_2plot*step)), patchsize_hres);
    
    h = [0.3 0.4 0.3];
    h = [0.5 0.5];
    recon = imfilter(recon, h);    
    recon = imfilter(recon, h');
%     h = fspecial('average', [3 3]);
%     recon = imfilter(recon, h);
    
%     h = fspecial('sobel');
%     reconSobelH = imfilter(recon, h);
%     reconSobelV = imfilter(recon, h');
    
    enhanceFrac = 0.05;
%     recon = recon + reconSobelH*enhanceFrac + reconSobelV*enhanceFrac;
    recon = normalize(recon);
    imshow(recon)
    title('Recon_HRes')

    subplot(2, 5, 3)
%     imshow(patch2im(Y(17:80,(r+1):(r+l)), patchsize_hres))
    imshow(...
        normalize(patch2im(...
        YL(:,(1 + (which_image_2plot-1)*step):(which_image_2plot*step)),...
        patchsize_lres))...
        );
    title('Actual_LRes')        

    subplot(2, 5, 4)
    recon = normalize(...
        patch2im(...
        Y_approx(1:patchsize_lres^2,(1 + (which_image_2plot-1)*step):(which_image_2plot*step)),...
        patchsize_lres)...
        );
    imshow(recon)
    title('Recon_LRes')
end
%% Gibbs
normalize = @(Mat) (Mat - min(Mat(:)))/(max(Mat(:)) - min(Mat(:)));
Pack_n_Show_dictionary(DH, DL);
figure(2)
clf

which_image_2plot = 2;
isAddingMean = false;
recon = patch2im(...
        Y, ...
        which_image_2plot, reduceTo_lres, reduceTo_lres,...
        patchsize_lres, isAddingMean*means_of_YL, overlap_low);
sres_bcubic = imresize(recon, 2, 'bicubic');

training_started=true;

tune_length = 10;
round_1 = tune_length;
round_2 = round_1 + tune_length;

mse_array = zeros(2000, 1);
for gr = 1:2000
    % Test here only
    Y_approxH = DH*(S.*B) + repmat(biasH, 1, c.N);
    Y_approx = D*(S.*B) + repmat(bias, 1, c.N);
    
    er = (sum((Y_approx(:) - Y(:)).^2))/(c.N*c.M);
    fprintf('-----------\n');
    fprintf('MSE: %10.8f\n', er);
    mse_array(gr) = er;
    
    step = (reduceTo_hres/patchsize_hres)^2;
    subplot(2, 5, 1)
    recon = patch2im(...
        YH, ...
        which_image_2plot, reduceTo_hres, reduceTo_hres,...
        patchsize_hres, isAddingMean*means_of_YH, overlap_high);
    imshow(recon);
    title('Actual_HRes')

    subplot(2, 5, 2)
    recon = patch2im(...
        Y_approxH, ...
        which_image_2plot, reduceTo_hres, reduceTo_hres,...
        patchsize_hres, isAddingMean*means_of_YL, overlap_high);
    imshow(recon)
    title('Recon_HRes')

    subplot(2, 5, 3)
%     imshow(patch2im(Y(17:80,(r+1):(r+l)), patchsize_hres))
    recon = patch2im(...
        Y, ...
        which_image_2plot, reduceTo_lres, reduceTo_lres,...
        patchsize_lres, isAddingMean*means_of_YL, overlap_low);
    imshow(recon);
    title('Actual_LRes')        

    subplot(2, 5, 4)
    recon = patch2im(...
        Y_approx, ...
        which_image_2plot, reduceTo_lres, reduceTo_lres,...
        patchsize_lres, isAddingMean*means_of_YL, overlap_low);
    imshow(recon)
    title('Recon_LRes')
    
    subplot(2, 5, 5)
    imagesc(B)
    title('B Matrix')
    
    subplot(2, 5, 6)
    imshow(sres_bcubic)
    title('Bicubic')
    
    drawnow
    
    tic
    % LEarn layer 1
    [ ~, S, B, PI, post_PI, ~, Gamma] = GibbsLevel( Y, D, S, B, PI, post_PI, bias, Gamma, Alpha1, Beta1, c, false );
    fprintf('[V1_L1]Iteration Complete: %d \n', gr)
    if layer2
        Y2 = sigmoid_Inv(post_PI);
        if reStructure
            Y2 = repatch(Y2, reduceTo, patchsize, K1, totalImages);
        end
    end
    if mod(gr, 2) == 0
        if sum(sum(B == 0)) == c.N*K1
            display('Resetting B1')
            [ ~, ~, B, ~, ~, ~, ~, ~ ] = InitAll( Y, K1, Alpha1, Beta1 );
        end
    end
    
    toc
    if layer2
        fprintf('Noise Var: L1 -> %3.4f, L2 -> %3.4f\n', 1/Gamma.n, 1/Gamma2.n)
    else
        fprintf('Noise Var: L1 -> %3.4f\n', 1/Gamma.n)
    end

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
Lf = eye(patchsize_hres^2);
% Lf = Lf(locate, :);

SNew = S;
% Hyperparameters
k1 = 1; % HyperParam for body similarity
k2 = 1; % HyperParam for similarity with low res image

% Define DHU and DHL
DHU = DH(1:(overlap_high*patchsize_hres), :);
DHL = DH((end - overlap_high*patchsize_hres + 1):end, :);
DHBody = DH(...
    (overlap_high*patchsize_hres):(end - overlap_high*patchsize_hres + 1), :);
LfDH = (Lf*DH);
% Calculate delJ
figure(4)
for iter_counter = 1:100
    if mod(iter_counter, 10) == 0
        fprintf('Iteration: %d\n',  iter_counter);
    end
    delJ = 0*S;
    for i = 2:size(S,2)
        delT1 = -2*(DHU'*DHL)*(SNew(:,i - 1).*B(:, i -1)) +...
            2*(DHU'*DHU)*(SNew(:,i).*B(:,i));
        delT2 = -2*k1*(DHBody'*DHBody)*(S(:,i).*B(:,i)) + ...
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