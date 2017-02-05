clear
im1 = imread('D:\ProjectData\caltech101\101_ObjectCategories\rhino\image_0001.jpg');
im1 = double(rgb2gray(im1))/255;

SZ = 64.0;
im1 = imresize(im1, SZ/255);
im1 = im1(1:SZ, 1:SZ);
im2 = upSample_with_noise(im1);
% im2 = upSample_with_noise(im2);
figure
imshow(im1)
figure
imshow(im2)
imwrite(im2, 'D:/ProjectData/caltech101/101_ObjectCategories/super_res_test/rhino_001.png')
%% (Dictionary) Coefficient learning follows
%% Load Layer 1
gcp
tic
load('D:\Users\Bishal Santra\Documents\MATLAB\MTP\neural_generative\WSs_10thSem\cat_6_img_12_beta1000.mat')
toc
%% Set folder
close all
imgPath = 'D:/ProjectData/caltech101/101_ObjectCategories/'
typeofimage = 'super_res_test/'
%% Matrix created here
reduceTo = SZ*2;
patchsize = 8;
column = 1;
totalImages = 1;
Y = GetDataMatrix([imgPath typeofimage], reduceTo, patchsize, totalImages);
%% 
figure(1)
clf
ii = 1;
step = size(Y,2)/totalImages;
recon = patch2im(Y(:,(1 + (ii-1)*step):(ii*step)), patchsize);
% recon = reshape(D(:, 23), reduceTo, reduceTo);
imshow(recon)
%% Initialize Layer 1

% K1 = 200;

Alpha1 = {};
Beta1 = {};

% Params for gamma distro - LAYER 1
Alpha1.d = 10;
Beta1.d = 10;
Alpha1.s = 10;
Beta1.s = 10;
Alpha1.bias = 10;
Beta1.bias = 10;
Alpha1.n = 1;
Beta1.n = 1;

% Params for beta distro : Near to zero, sparse
Alpha1.pi = 1;
Beta1.pi = 2000;

tic
[ ~, S, B, PI, post_PI, bias, Gamma, c ] = InitAll( Y, K1, Alpha1, Beta1 );
toc
%% Initialize Layer 2

% Whether training layer 2
layer2 = false;
reStructure = false;

K2 = 100;

% LAYER 2 Settings
Alpha2 = Alpha1;
Beta2 = Beta1;

Alpha2.d = 1e-1;
Beta2.d = 1e-1;
Alpha2.s = 1e-1;
Beta2.s = 1e-1;

Alpha2.pi = 1;
Beta2.pi = 2000;
Alpha1.n = 1e-3;
Beta1.n = 1e-3;

if(layer2)
    tic
%     Y2 = sigmoid_Inv(post_PI);
    Y2 = S.*B;
    if reStructure
        Y2 = repatch_v2(Y2, reduceTo, patchsize, K1, totalImages);
    end
    [ D2, S2, B2, PI2, post_PI2, bias2, Gamma2, c2 ] = InitAll( Y2, K2, Alpha2, Beta2 );
    toc
end
%% Gibbs
figure(2)
clf

tune_length = 10;
round_1 = tune_length;
round_2 = round_1 + tune_length;

mse_array = zeros(2000, 1);
for gr = 1:2000
    % Test here only
    Y_approx = D*(S.*B) + repmat(bias, 1, c.N);
    if ~layer2
        er = (sum((Y_approx(:) - Y(:)).^2))/(c.N*c.M);
        fprintf('-----------\n');
        fprintf('MSE: %6.3f\n', er);
        mse_array(gr) = er;
    else
        er = (sum((Y_approx(:) - Y(:)).^2))/(c.N*c.M);
        Y2_approx = D2*(S2.*B2) + repmat(bias2, 1, c2.N);
        er2 = (sum((Y2_approx(:) - Y2(:)).^2))/(c2.N*c2.M);
        fprintf('-----------\n');
        fprintf('MSE@1: %6.3f\n', er);
        fprintf('MSE@2: %6.3f\n', er2);
    end
    l = (reduceTo - patchsize + 1)^2;
    
    r = 0;
    subplot(2, 2, 1)
    imshow(patch2im(Y(:,(r+1):(r+l)), patchsize))
    title('Actual')        

    subplot(2, 2, 2)
    recon = patch2im(Y_approx(:,(r+1):(r+l)), patchsize);
    imshow(recon)
    title('Recon')

    subplot(2, 2, 3)
    imagesc(B)
    title('B Matrix')
    if(layer2)
        subplot(3, 3, 4)
        imagesc(Y2)
        title('Y2')
        
        subplot(3,3,5)
        imagesc(B2)
        title('Layer2 B2')
        
        subplot(3,3,6)
        imagesc(post_PI2)
        title('Post PI2')
        
        subplot(3,3,7)
        imagesc(Y2_approx)
        title('Recon Y2')
        
        subplot(3,3,8)
        imagesc(S2.*B2)
        title('S2.*B2')
    end
    
    drawnow
    
    tic
    
%     if mod(gr, 2) == 1 || ~layer2
    if ~layer2
        % LEarn layer 1
        [ ~, S, B, PI, post_PI, bias, Gamma] = GibbsLevel( Y, D, S, B, PI, post_PI, bias, Gamma, Alpha1, Beta1, c, false );        
        fprintf('[V1_L1]Iteration Complete: %d \n', gr)
        if layer2
            Y2 = sigmoid_Inv(post_PI);
            if reStructure
                Y2 = repatch(Y2, reduceTo, patchsize, K1, totalImages);
            end
        end
    else
        %Learn Layer 2
        [D2, S2, B2, PI2, post_PI2, bias2, Gamma2] = GibbsLevel( Y2, D2, S2, B2, PI2, post_PI2, bias2, Gamma2, Alpha2, Beta2, c2 );
        fprintf('[V1_L2]Iteration Complete: %d \n', gr)
    end
%         % Learn Both
%         [ D, S, B, PI, post_PI, bias, Gamma] = GibbsLevel( Y, D, S, B, PI, post_PI, bias, Gamma, Alpha1, Beta1, c );
%         Y2 = sigmoid_Inv(post_PI);
%         [D2, S2, B2, PI2, post_PI2, bias2, Gamma2] = GibbsLevel( Y2, D2, S2, B2, PI2, post_PI2, bias2, Gamma2, Alpha2, Beta2, c2 );
%     end
    
    % save('WSs/runtime_backup_extend_16p', '-v7.3');
    % Checkpoint for B - Layer 1
    if mod(gr, 2) == 0
        if sum(sum(B == 0)) == c.N*K1
            display('Resetting B1')
            [ ~, ~, B, ~, ~, ~, ~, ~ ] = InitAll( Y, K1, Alpha1, Beta1 );
        end
    end
    
    % Checkpoint for B - Layer 2
    if mod(gr, 5) == 0 && layer2
        if sum(sum(B2 == 0)) == c2.N*K2
            display('Resetting B2')
            [ ~, ~, B2, ~, ~, ~, ~, ~ ] = InitAll( Y2, K2, Alpha2, Beta2 );
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
