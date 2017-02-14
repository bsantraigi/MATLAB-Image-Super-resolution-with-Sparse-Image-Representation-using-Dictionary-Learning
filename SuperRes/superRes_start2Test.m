%% (Dictionary) Coefficient learning follows
%% Load Layer 1
gcp
tic
load('D:\Users\Bishal Santra\Documents\MATLAB\MTP\SuperRes\WSs_10thSem\8_n_16_coupled_v3_mean_subbed.mat')
toc
DL = D(1:16, :);
DH = D(17:end, :);
DFull = D;
D = DL;

biasFull = bias;
biasL = bias(1:16);
biasH = bias(17:end);
bias = biasL;
%% Set folder
close all
imgPath = './'
typeofimage = 'super_res_test/'
%% Matrix created here
close all
reduceTo_lres = 128;
reduceTo_hres = 256;
patchsize_lres = 4;
patchsize_hres = 8;
column = 1;
totalImages = 2;
YL = GetDataMatrix([imgPath 'lres/'], reduceTo_lres, patchsize_lres, totalImages);
YH = GetDataMatrix([imgPath 'hres/'], reduceTo_hres, patchsize_hres, totalImages);
Y = [YL];
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
[ ~, S, B, PI, post_PI, ~, Gamma, c ] = InitAll( Y, K1, Alpha1, Beta1 );
toc

%% Plot images
training_started=false;
figure(1)
clf
r = 1;
subplot(2, 5, 1)
imshow(...
    normalize(...
    patch2im(YH(:,(1 + (r-1)*step):(r*step)), patchsize_hres)));
title('Actual_HRes')        

subplot(2, 5, 3)
imshow(...
    normalize(...
    patch2im(...
    YL(:,(1 + (r-1)*step):(r*step)),...
    patchsize_lres))...
    );
title('Actual_LRes') 

%% Plot recon
if training_started
    figure(2)
    clf
    r = 1;
    subplot(2, 5, 1)
%     imshow(patch2im(Y(17:80,(r+1):(r+l)), patchsize_hres))
    imshow(...
        normalize(...
        patch2im(YH(:,(1 + (r-1)*step):(r*step)), patchsize_hres))...        
    );
    title('Actual_HRes')

    subplot(2, 5, 2)
    recon = patch2im(Y_approxH(:,(1 + (r-1)*step):(r*step)), patchsize_hres);
    
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
        YL(:,(1 + (r-1)*step):(r*step)),...
        patchsize_lres))...
        );
    title('Actual_LRes')        

    subplot(2, 5, 4)
    recon = normalize(...
        patch2im(...
        Y_approx(1:patchsize_lres^2,(1 + (r-1)*step):(r*step)),...
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
    
    r = 1;
    step = (reduceTo_hres/patchsize_hres)^2;
    subplot(2, 5, 1)
%     imshow(patch2im(Y(17:80,(r+1):(r+l)), patchsize_hres))
    recon = normalize(patch2im(YH(:,(1 + (r-1)*step):(r*step)), patchsize_hres));
    imshow(recon);
    title('Actual_HRes')

    subplot(2, 5, 2)
    recon = normalize(patch2im(Y_approxH(:,(1 + (r-1)*step):(r*step)), patchsize_hres));
    imshow(recon)
    title('Recon_HRes')

    subplot(2, 5, 3)
%     imshow(patch2im(Y(17:80,(r+1):(r+l)), patchsize_hres))
    recon = normalize(...
        patch2im(...
        YL(:,(1 + (r-1)*step):(r*step)),...
        patchsize_lres)...
        );
    imshow(recon);
    title('Actual_LRes')        

    subplot(2, 5, 4)
    recon = normalize(...
        patch2im(...
        Y_approx(1:patchsize_lres^2,(1 + (r-1)*step):(r*step)),...
        patchsize_lres...
        ));
    
    imshow(recon)
    title('Recon_LRes')
    
    subplot(2, 5, 5)
    imagesc(B)
    title('B Matrix')
    
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