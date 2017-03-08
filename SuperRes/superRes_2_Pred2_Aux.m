clear
close all
% pathRoot = 'D:/ProjectData/caltech101/101_ObjectCategories/super_res_test/';
imcategory = 'llama';
pname = 'image_0020.jpg';
im1 = imread(['D:\ProjectData\caltech101\101_ObjectCategories\' imcategory '\' pname]);
if size(im1, 3) > 1
im1 = double(rgb2gray(im1))/255;
else
    im1 = double(im1)/255;
end
figure
imshow(im1)
L = min(size(im1));
SZ = 128.0;
if(L > SZ)
    im1 = imresize(im1, SZ/L);
    
    im1 = im1(1:SZ, 1:SZ);
    im2 = imresize(im1, 0.5);
    % im2 = upSample_with_noise(im2);
    figure
    imshow(im1)
    figure
    imshow(im2)
    imwrite(im1, ['hres/', pname])
    imwrite(im2, ['lres/' pname])
else
    fprintf('Image too small\n')
end
%% (Dictionary) Coefficient learning follows
%% Load Layer 1
gcp
tic
% load('D:\Users\Bishal Santra\Documents\MATLAB\MTP\neural_generative\WSs_10thSem\cat_6_img_12_beta1000.mat')
toc
%% Set folder
imgPath = './'
typeofimage = 'super_res_test/'

%% Matrix created here
close all
reduceTo_lres = 64;
reduceTo_hres = 128;
patchsize_lres = 4;
patchsize_hres = 8;
column = 1;
totalImages = 2;
overlap_low = 1;
overlap_high = 2;
[YL, means_of_YL] = GetDataMatrix([imgPath 'lres/'],...
    reduceTo_lres, patchsize_lres, totalImages, overlap_low);
[YH, means_of_YH] = GetDataMatrix_4x([imgPath 'hres/'],...
    reduceTo_hres, patchsize_hres, totalImages, overlap_high);
% Y = [YL; YH];
% means_of_Y = [means_of_YL; means_of_YH];

%% Initialize Layer 1
close all
% K1 = 400;
% 
% Alpha1 = {};
% Beta1 = {};
% 
% % Params for gamma distro - LAYER 1
% Alpha1.d = 1;
% Beta1.d = 1;
% Alpha1.s = 1e-1;
% Beta1.s = 1e-1;
% Alpha1.bias = 1e-1;
% Beta1.bias = 1e-1;
% Alpha1.n = 1e-3;
% Beta1.n = 1e-3;
% 
% % Params for beta distro : Near to zero, sparse
% Alpha1.pi = 1;
% Beta1.pi = 800;

tic
[ ~, ~, S, B, ~, post_PI, ~, ~, ~, c ] =...
    InitAll( YH, YL, K1, Alpha1, Beta1 );
toc
%% Plot images
figure(1)
clf
r = 5;
step = (reduceTo_hres/patchsize_hres)^2;
subplot(2, 5, 1)
recon = patch2im(...
    Y((1+patchsize_lres^2):end, :), ...
    r, reduceTo_hres, reduceTo_hres,...
    patchsize_hres, means_of_YH, overlap_high);
imshow(recon);
title('Actual_HRes')

subplot(2, 5, 2)
recon = patch2im(...
    Y_approx((1+patchsize_lres^2):end, :), ...
    r, reduceTo_hres, reduceTo_hres,...
    patchsize_hres, means_of_YH, overlap_high);
imshow(recon)
title('Recon_HRes')

subplot(2, 5, 3)
%     imshow(patch2im(Y(17:80,(r+1):(r+l)), patchsize_hres))
recon = patch2im(...
    Y(1:patchsize_lres^2, :), ...
    r, reduceTo_lres, reduceTo_lres,...
    patchsize_lres, means_of_YL, overlap_low);
imshow(recon);
title('Actual_LRes')        

subplot(2, 5, 4)
recon = patch2im(...
    Y_approx(1:patchsize_lres^2, :), ...
    r, reduceTo_lres, reduceTo_lres,...
    patchsize_lres, means_of_YL, overlap_low);
imshow(recon)
title('Recon_LRes')
%% Gibbs
figure(2)
clf

tune_length = 10;
round_1 = tune_length;
round_2 = round_1 + tune_length;

mse_array = zeros(2000, 1);

isAddingMean = true;
for gr = 1:2000
    % Test here only
    YH_approx = DH*(S.*B) + repmat(biasH, 1, c.N);
    YL_approx = DL*(S.*B) + repmat(biasL, 1, c.N);
    
    erH = (sum((YH_approx(:) - YH(:)).^2))/(c.N*c.MH);
    erL = (sum((YL_approx(:) - YL(:)).^2))/(c.N*c.ML);
    fprintf('-----------\n');
    fprintf('MSE: %10.8f, %10.8f\n', erH, erL);
    mse_array(gr) = erH;
    
    r = 1;
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
    
    drawnow
    
    tic    
    % LEarn layer 1
    [ ~, ~, S, B, ~, post_PI, ~, ~, ~ ] = ...
        GibbsLevel( YH, YL, DH, DL, S, B, PI, post_PI,...
        biasH, biasL, Gamma, Alpha1, Beta1, c );        
    fprintf('[V1_L1]Iteration Complete: %d \n', gr)
    
    if mod(gr, 2) == 0
        if sum(sum(B == 0)) == c.N*K1
            display('Resetting B1')            
            [ ~, ~, ~, B, ~, ~, ~, ~, ~, ~ ]...
                = InitAll( YH, YL, K1, Alpha1, Beta1 );
        end
    end

    toc
    fprintf('Noise Vars: %3.4f, %3.4f\n', 1/Gamma.nH, 1/Gamma.nL)

end
fprintf('Gibbs Complete...\n')
%% Show dictionaries in order
usageIndex = sum(B, 2);
[~, order_in] = sort(usageIndex, 'descend');
Pack_n_Show_dictionary(DH(:, order_in), DL(:, order_in))
figure(198)
plot(PI(order_in));
%% Plot reconstructed Image
figure(2)
clf
Y_approx = D*(S.*B) + repmat(bias, 1, c.N);
l = (reduceTo - patchsize + 1)^2;
for r = 0:l:(c.N - 1)
    subplot(1, 2, 1)
    recon = patch2im(Y_approx(:,(r+1):(r+l)), patchsize);
    recon(recon<=0) = 0;
    recon(recon>=1) = 1;   
    
    imshow(recon)    
    title('Recon')
    subplot(1, 2, 2)
    actual = patch2im(Y(:,(r+1):(r+l)), patchsize);
    
%     cl = clock;
%     suffix = sprintf('%d%d%d%d', cl(3), cl(4), cl(5), floor(10*cl(6)));
%     imwrite(recon, sprintf('outputs_nov/recon_%s.png', suffix));
%     imwrite(actual, sprintf('outputs_nov/actual_%s.png', suffix));
    
    imshow(actual)
    title('Actual')
    drawnow
    pause(0.4)
end
%% Plot Mse
figure(1)
clf
plot(mse_array(1:gr))
axis([1 gr 0 1.5])
xlabel('Iteration')
ylabel('MSE')
title('MSE in approximation as a function of Gibbs Iteration')
dim = [.2 0.3 .3 0];
str = 'For training of 128 x 128 images, 8 x 8 patchsize';
annotation('textbox',dim,'String',str,'FitBoxToText','on');
%% Plot reconstructed Image 2
figure(2)
clf
Y2_app = D2*(S2.*B2) + repmat(bias2, 1, c2.N);
new_pi = 1./(1+exp(-Y2_app));
B_new = binornd(ones(K1, c.N), new_pi);
Y_approx = D*(S.*B_new) + repmat(bias, 1, c.N);
Y_approx_1 = D*(S.*B) + repmat(bias, 1, c.N);
l = (reduceTo - patchsize + 1)^2;
for r = 0:l:2500
    subplot(1, 2, 1)
    recon = patch2im(Y_approx(:,(r+1):(r+l)), patchsize);
    recon(recon<=0) = 0;
    recon(recon>=1) = 1;
    imshow(recon)
    title('Recon')
    subplot(1, 2, 2)
    imshow(patch2im(Y_approx_1(:,(r+1):(r+l)), patchsize))
    title('Actual')
    drawnow
    pause(0.3)
end
%% Plot the sorted B matrix
[~, bi] = sort(sum(B, 2));
figure(5), imagesc(B(bi, :))
%% Plot the sorted B2 matrix
[~, bi] = sort(sum(B2, 2));
figure(5), imagesc(B2(bi, :))
%% Plot all features
normalize = @(mat) (mat - min(min(mat)))/(max(max(mat)) - min(min(mat)));
% muD1_new = normalize(muD1);
muD1_new = D;
figure(2)
clf
gridsize = 5;
l = (reduceTo - patchsize + 1)^2;
subplot(gridsize, gridsize, 1)
Y_approx = D*(S.*B) + repmat(bias, 1, c.N);
imshow(patch2im(Y_approx(:,1:l), patchsize))
sb = 2;
[~, list_of_f] = sort(sum(B,2));
list_of_f = list_of_f(end:-1:(end - 11));
for i = 1:length(list_of_f)
    sb = i*2;
%     active_f = fs(sb - 1);
    active_f = list_of_f(i);
    tempB = B;
    tempB([1:(active_f - 1), (active_f+1):K1], :) = 0;
    Y_approx = D*(S.*tempB) + repmat(bias, 1, c.N);
    
    subplot(gridsize, gridsize, sb)
    recon = normalize(patch2im(Y_approx(:,1:l), patchsize));
    imshow(recon)
    title(sprintf('Feature %d', active_f))
    
%     cl = clock;
%     suffix = sprintf('%d%d%d%d', cl(3), cl(4), cl(5), floor(10*cl(6)));
%     imwrite(recon, sprintf('outputs_nov/feat_%d.png', active_f));
    
    if totalImages > 1
        subplot(gridsize, gridsize, sb + 1)
        recon = normalize(patch2im(Y_approx(:,(l + 1):2*l), patchsize));
        imshow(recon)
        title(sprintf('Feature %d', active_f))
    end
    
end
%% Plot higher level features
normalize = @(mat) (mat - min(min(mat)))/(max(max(mat)) - min(min(mat)));
% muD1_new = normalize(muD1);
figure(2)
clf
gridsize = 5;
l = (reduceTo - patchsize + 1)^2;
subplot(gridsize, gridsize, 1)
Y_approx = D*(S.*B) + repmat(bias, 1, c.N);
imshow(patch2im(Y_approx(:,1:l), patchsize))
sb = 2;
[~, list_of_f] = sort(sum(B2,2));
list_of_f = list_of_f(end:-1:(end - 11));
for i = 1:length(list_of_f)
    sb = i*2;
%     active_f = fs(sb - 1);
    active_f = list_of_f(i);

    Y2_approx = D2(:, active_f)*(S2(active_f, :).*B2(active_f, :)) + repmat(bias2, 1, c.N);
    Y2_approx = 1./(1+exp(-Y2_approx));
%     tempB = Y2_approx > 0.5;
    tempB = binornd(ones(K1, c.N), Y2_approx);
    Y_approx = D*(S.*(Y2_approx.*B)) + repmat(bias, 1, c.N);
    
    subplot(gridsize, gridsize, sb)
    recon = (patch2im(Y_approx(:,1:l), patchsize));
    imshow(recon)
    title(sprintf('Feature %d', active_f))
    if totalImages > 1
    subplot(gridsize, gridsize, sb + 1)
    recon = (patch2im(Y_approx(:,(l + 1):2*l), patchsize));
    imshow(recon)
    title(sprintf('Feature %d', active_f))
    end
end

%% Draw Lower Level Patches
figure(5)
clf

% Best Features same location
[~, list_of_f] = sort(sum(B,2));
list_of_f = list_of_f(end:-1:(end - 24));
for i = 1:8
    subplot(5,5,i)
    j = list_of_f(i);
    imshow(reshape(D(:, j), patchsize, patchsize))
    imagesc(reshape(D(:, j), patchsize, patchsize)); colormap;
    
    imwrite(normalize(reshape(D(:, j), patchsize, patchsize)), sprintf('outputs_nov/patch_%d.png', j));
    
    title(sprintf('Feature %d', j))
end
figure(6)
clf
% Same Features Different Patches
[~, list_of_f] = sort(sum(B,2));
list_of_f = list_of_f(end:-1:(end - 24));
j = list_of_f(4);
Y_approx = D(:, j)*(S(j, :).*B(j,:)) + repmat(bias, 1, c.N);

for i = 1:25
    subplot(5,5,i)
    imshow(reshape(Y_approx(:, 8*i), patchsize, patchsize))
    imagesc(reshape(Y_approx(:, 8*i), patchsize, patchsize)); colormap;    
    title(sprintf('Patch %d', 8*i))
end
%% Draw Higher Level Patches

figure(7)
clf
% Best Features same location
[~, list_of_f] = sort(sum(B2,2));
list_of_f = list_of_f(end:-1:(end - 24));
for i = 1:25
    subplot(5,5,i)
    j = list_of_f(i);
    b_temp = 1./(1 + exp(-D2(:, j)*(S2(j, :).*B2(j, :)) + repmat(bias2, 1, c2.N)));
    b_temp = binornd(ones(K1, c.N), b_temp);
    Y_approx = D*(S.*b_temp.*B) + repmat(bias, 1, c.N);
    imshow(reshape(Y_approx(:, 10), patchsize, patchsize))
    imagesc(reshape(Y_approx(:, 10), patchsize, patchsize)); colormap;
    title(sprintf('Feature %d', j))
end

figure(8)
clf

% Same Features Different Patches
[~, list_of_f] = sort(sum(B2,2));
list_of_f = list_of_f(end:-1:(end - 24));
j = list_of_f(1);
b_temp = 1./(1 + exp(-D2(:, j)*(S2(j, :).*B2(j, :)) + repmat(bias2, 1, c2.N)));
b_temp = binornd(ones(K1, c.N), b_temp);
Y_approx = D*(S.*b_temp.*B) + repmat(bias, 1, c.N);
for i = 1:25
    subplot(5,5,i)
    imshow(reshape(Y_approx(:, 8*i), patchsize, patchsize))
    imagesc(reshape(Y_approx(:, 8*i), patchsize, patchsize)); colormap;
    title(sprintf('Patch %d', 8*i))
end
