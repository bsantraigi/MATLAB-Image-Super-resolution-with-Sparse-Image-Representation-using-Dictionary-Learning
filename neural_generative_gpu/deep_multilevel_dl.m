%% Actual Code
clear all
close all
imgPath = 'D:/ProjectData/caltech101/101_ObjectCategories/'
typeofimage = 'Faces_easy/'
fl = dir([imgPath typeofimage]);
%%
% Matrix created here
% Y
Y = [];
reduceTo = 64;
patchsize = 8;
column = 1;
totalImages = 2;
gap = 7;
for imindex = 3:gap:(3 + gap*totalImages - 1)    
    imgTemp = imread([imgPath typeofimage fl(imindex).name]);
    if size(imgTemp, 3) > 1
        imgTemp = rgb2gray(imgTemp);
    end
    imgTemp = imresize(imgTemp, reduceTo/min(size(imgTemp)));
    if size(imgTemp, 1) > reduceTo
        r = size(imgTemp, 1);
        drop = floor((r - reduceTo)/2);
        s = max(1, 1+drop - 1);
        imgTemp = imgTemp(s:(s+reduceTo-1),:);
    elseif size(imgTemp, 2) > reduceTo
        r = size(imgTemp, 2);
        drop = floor((r - reduceTo)/2);
        s = max(1, 1+drop - 1);
        imgTemp = imgTemp(:,s:(s+reduceTo-1));
    end
    Y = [Y im2patch(imgTemp, patchsize)];
    size(imgTemp)
    imshow(imgTemp)
    drawnow
    pause(0.3)
end
Y = Y./255;
Y = gpuArray(Y);
clearvars r s drop
%%
figure(1)
clf
ii = 2;
step = size(Y,2)/totalImages;
recon = patch2im(Y(:,(1 + (ii-1)*step):(ii*step)), patchsize);
% recon = reshape(D(:, 23), reduceTo, reduceTo);
imshow(recon)
%% Initialize
K1 = 80;
K2 = 80;

Alpha1 = {};
Beta1 = {};

% Params for gamma distro
Alpha1.d = 1e-1;
Beta1.d = 1e-1;
Alpha1.s = 1e-1;
Beta1.s = 1e-1;
Alpha1.bias = 1e-1;
Beta1.bias = 1e-1;
Alpha1.n = 1e-3;
Beta1.n = 1e-3;

% Params for beta distro : Near to zero, sparse
Alpha1.pi = 1;
Beta1.pi = 800;

Alpha2 = Alpha1;
Beta2 = Beta1;

Alpha2.pi = 1;
Beta2.pi = 2000;

tic
[ D, S, B, PI, post_PI, bias, Gamma, c ] = InitAll( Y, K1, Alpha1, Beta1 );
toc
layer2 = false;
if(layer2)
    Y2 = sigmoid_Inv(post_PI);
    [ D2, S2, B2, PI2, post_PI2, bias2, Gamma2, c2 ] = InitAll( Y2, K2, Alpha2, Beta2 );
end
%% Gibbs
figure(2)
clf

tune_length = 10;
round_1 = tune_length;
round_2 = round_1 + tune_length;

mse_array = gpuArray(zeros(2000, 1));
for gr = 1:2000
    % Test here only
    Y_approx = D*(S.*B) + repmat(bias, 1, c.N);
    er = (sum((Y_approx(:) - Y(:)).^2))/c.N
    mse_array(gr) = er;
    l = (reduceTo - patchsize + 1)^2;
    for r = 0:l:0
        subplot(3, 2, 1)
        imshow(patch2im(Y(:,(r+1):(r+l)), patchsize))
        title('Actual')        
        
        subplot(3, 2, 2)
        recon = patch2im(Y_approx(:,(r+1):(r+l)), patchsize);
        imshow(recon)
        title('Recon')
        
        subplot(3, 2, 3)
        imagesc(B)
        title('B Matrix')
        
        if(layer2)
            subplot(3, 2, 4)
            imagesc(Y2)
            title('Y2')
        end
    end
    if layer2
        subplot(3,2,5)
        imagesc(B2)
        title('Layer2 B2')
        subplot(3,2,6)
        imagesc(post_PI2)
        title('Post PI2')
        drawnow
    end
    
    tic
    if mod(floor(gr/10), 2) == 0 || true
        % LEarn layer 1
        [ D, S, B, PI, post_PI, bias, Gamma] = GibbsLevel( Y, D, S, B, PI, post_PI, bias, Gamma, Alpha1, Beta1, c );
        %Y2 = sigmoid_Inv(post_PI);
        fprintf('[V1_L1]Iteration: %d \n', gr)
    else
        %Learn Layer 2
%         [D2, S2, B2, PI2, post_PI2, bias2, Gamma2] = GibbsLevel( Y2, D2, S2, B2, PI2, post_PI2, bias2, Gamma2, Alpha2, Beta2, c2 );
%         fprintf('[V1_L2]Iteration: %d \n', gr)
    end
%         % Learn Both
%         [ D, S, B, PI, post_PI, bias, Gamma] = GibbsLevel( Y, D, S, B, PI, post_PI, bias, Gamma, Alpha1, Beta1, c );
%         Y2 = sigmoid_Inv(post_PI);
%         [D2, S2, B2, PI2, post_PI2, bias2, Gamma2] = GibbsLevel( Y2, D2, S2, B2, PI2, post_PI2, bias2, Gamma2, Alpha2, Beta2, c2 );
%     end
    
    save('WSs/runtime_backup_extend_16p', '-v7.3');
    % Checkpoint for B - Layer 1
    if mod(gr, 2) == 0
        if sum(sum(B == 0)) == c.N*K1
            display('Resetting B1')
            [ ~, ~, B, ~, ~, ~, ~, ~ ] = InitAll( Y, K1, Alpha1, Beta1 );
        end
    end
    
    % Checkpoint for B - Layer 2
    if mod(gr, 5) == 0 && layer2
        if sum(sum(B2 == 0)) == c.N*K2
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
    pause(1)
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