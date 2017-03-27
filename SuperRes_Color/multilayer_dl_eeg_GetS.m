%% Actual Code
clear all
close all
user = 1;
sData = load(sprintf('EEG_Kaggle/user%d_Dictionaries_set1', user));
Y = LoadEEG_TestData(user);
% Y = LoadEEG_TrainData(1, 1, 2000, 'interictal');
imagesc(Y);
%% Initialize
K1 = 60;
K2 = 60;

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
Beta1.pi = 2500;

Alpha2 = Alpha1;
Beta2 = Beta1;

Alpha2.pi = 1;
Beta2.pi = 2500;
D = sData.D;
D2 = sData.D2;
[ ~, S, B, PI, post_PI, bias, Gamma, c ] = InitAll( Y, K1, Alpha1, Beta1 );
Y2 = sigmoid_Inv(post_PI);
[ ~, S2, B2, PI2, post_PI2, bias2, Gamma2, c2 ] = InitAll( Y2, K2, Alpha2, Beta2 );

%% Gibbs
figure(2)
clf

tune_length = 10;
round_1 = tune_length;
round_2 = round_1 + tune_length;

for gr = 1:2000
    % Test here only
    Y_approx = D*(S.*B) + repmat(bias, 1, c.N);
    
    subplot(3, 2, 1)
    imagesc(Y_approx)
    title('Recon1')

    subplot(3, 2, 2)
    imagesc(Y)
    title('Actual1')
    
    subplot(3, 2, 3)
    imagesc(B)
    title('B Matrix')

    subplot(3, 2, 4)
    imagesc(S.*B)
    title('S.*B')
    
    subplot(3,2,5)
    imagesc(B2)
    title('Layer2 B2')
    
    subplot(3,2,6)
    imagesc(S2.*B2)
    title('S2.*B2')
    drawnow
    
    tic
    if mod(floor(gr/20), 2) == 0
        % LEarn layer 1
        [ ~, S, B, PI, post_PI, bias, Gamma] = GibbsLevel( Y, D, S, B, PI, post_PI, bias, Gamma, Alpha1, Beta1, c, 0 );
        Y2 = sigmoid_Inv(post_PI);
        fprintf('[V1_L1]Iteration: %d \n', gr)
    else
        %Learn Layer 2
        [~, S2, B2, PI2, post_PI2, bias2, Gamma2] = GibbsLevel( Y2, D2, S2, B2, PI2, post_PI2, bias2, Gamma2, Alpha2, Beta2, c2, 0 );
        fprintf('[V1_L2]Iteration: %d \n', gr)
    end
%         % Learn Both
%         [ D, S, B, PI, post_PI, bias, Gamma] = GibbsLevel( Y, D, S, B, PI, post_PI, bias, Gamma, Alpha1, Beta1, c );
%         Y2 = sigmoid_Inv(post_PI);
%         [D2, S2, B2, PI2, post_PI2, bias2, Gamma2] = GibbsLevel( Y2, D2, S2, B2, PI2, post_PI2, bias2, Gamma2, Alpha2, Beta2, c2 );
%     end
    
%     save('WSs/runtime_backup_extend_16p', '-v7.3');
    % Checkpoint for B - Layer 1
    if mod(gr, 2) == 0
        if sum(sum(B == 0)) == c.N*K1
            display('Resetting B1')
            [ ~, ~, B, ~, ~, ~, ~, ~ ] = InitAll( Y, K1, Alpha1, Beta1 );
        end
    end
    
    % Checkpoint for B - Layer 2
    if mod(gr, 5) == 0
        if sum(sum(B2 == 0)) == c.N*K2
            display('Resetting B2')
            [ ~, ~, B2, ~, ~, ~, ~, ~ ] = InitAll( Y2, K2, Alpha2, Beta2 );
        end
    end

    toc
    
    fprintf('Noise Var: L1 -> %3.4f, L2 -> %3.4f\n', 1/Gamma.n, 1/Gamma2.n)
%     if(1/Gamma2.n < 0.1)
%         break
%     end
end
fprintf('Gibbs Complete...\n')
%% Save S.*B for test matrix
fprintf('Save to -> EEG_Kaggle/test_data_user%d\n', user);
save(sprintf('EEG_Kaggle/test_data_user%d', user), '-v7.3')
%% Plot reconstructed Image
figure(2)
clf
Y_approx = D*(S.*B) + repmat(bias, 1, c.N);
l = (reduceTo - patchsize + 1)^2;
for r = 0:l:2500
    subplot(1, 2, 1)
    recon = patch2im(Y_approx(:,(r+1):(r+l)), patchsize);
    recon(recon<=0) = 0;
    recon(recon>=1) = 1;
    imshow(recon)
    title('Recon')
    subplot(1, 2, 2)
    imshow(patch2im(Y(:,(r+1):(r+l)), patchsize))
    title('Actual')
    drawnow
    pause(0.3)
end
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
for i = 1:25
    subplot(5,5,i)
    j = list_of_f(i);
    imshow(reshape(D(:, j), patchsize, patchsize))
    imagesc(reshape(D(:, j), patchsize, patchsize)); colormap;
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