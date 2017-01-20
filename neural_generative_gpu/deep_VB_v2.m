%% Important Image Pre-processing steps used
% 1. Normalize and make image 0 mean
% 2. While redrawing normalize the data to fall in the range [0, 1]
% 3. Good results when alpha.pi = 10 and beta.pi = 20
% 4. Don't set initial values of muD and muS to be zero
%% Actual Code
clear
normalize = @(mat) (mat - min(min(mat)))/(max(max(mat)) - min(min(mat)));
imgPath = 'caltech101/101_ObjectCategories/'
typeofimage = 'Faces_easy/'
fl = dir([imgPath typeofimage]);

Y = [];
reduceTo = 64;
patchsize = 5;
column = 1;
for imindex = 3:100:110
    imgTemp = rgb2gray(imread([imgPath typeofimage fl(imindex).name]));
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
    imgTemp = double(imgTemp);
    imgTemp = imgTemp./255;
    imgTemp = imgTemp - 0.5;
    Y = [Y im2patch(imgTemp, patchsize)];
%     size(imgTemp)
%     imshow(imgTemp)
%     drawnow
%     pause(0.3)
end
clearvars r s drop
%%
figure(1)
clf

recon = normalize(patch2im(Y(:,1:size(Y,2)/2), patchsize));
% recon = reshape(D(:, 23), reduceTo, reduceTo);
imshow(recon)
%% Initialize
K1 = 80;
K2 = 20;
N = size(Y, 2);
% Params for gamma and beta priors
% Params for beta distro : Near to zero, sparse
% Most sparse - 1.5 and 11
Alpha1 = struct('d', 3, 's', 3, 'n', 3, 'pi', 1);
Beta1 = struct('d', 0.3, 's', 0.3, 'n', 0.3, 'pi', 10);

Alpha2 = struct('d', 3, 's', 3, 'n', 3, 'pi', 1);
Beta2 = struct('d', 0.3, 's', 0.3, 'n', 0.3, 'pi', 15);

pi2y = @(PI_Mat) log(PI_Mat./(1-PI_Mat));

[ muD1, muS1, PI1, Gamma1, Palpha1, Pbeta1, c1 ] = InitAll_VB( Y, K1, Alpha1, Beta1 );
[ muD2, muS2, PI2, Gamma2, Palpha2, Pbeta2, c2 ] = InitAll_VB( pi2y(PI1), K2, Alpha2, Beta2 );


%% Gibbs
figure(2)
clf

for gr = 1:120
    tic
    % L1
    [ muD1, muS1, PI1, Gamma1, Palpha1, Pbeta1 ] = VB_Update(Y, muD1, muS1, PI1, Gamma1, Palpha1, Pbeta1, Alpha1, Beta1, c1 );
    
    % L2
    Y2 = pi2y(PI1);
    [ muD2, muS2, PI2, Gamma2, Palpha2, Pbeta2 ] = VB_Update( Y2, muD2, muS2, PI2, Gamma2, Palpha2, Pbeta2, Alpha2, Beta2, c2 );
    
    toc
    fprintf('Iteration: %d \n', gr)
    
    % Test here only
    Y_approx = muD1*(muS1.*PI1);
    l = (reduceTo - patchsize + 1)^2;
    for r = 0:l:1
        subplot(1, 2, 1)
        recon = normalize(patch2im(Y_approx(:,(r+1):(r+l)), patchsize));
        imshow(recon)
        title('Recon')
        subplot(1, 2, 2)
        imshow(normalize(patch2im(Y(:,(r+1):(r+l)), patchsize)))
        title('Actual')
        drawnow
    end
end
fprintf('Gibbs Complete...\n')
%%
figure(2)
clf
Y_approx = muD1*(muS1.*PI1);
l = (reduceTo - patchsize + 1)^2;
for r = 0:l:1.5*l
    subplot(1, 2, 1)
    recon = normalize(patch2im(Y_approx(:,(r+1):(r+l)), patchsize));
    recon(recon<=0) = 0;
    recon(recon>=1) = 1;
    imshow(recon)
    title('Recon')
    subplot(1, 2, 2)
    imshow(normalize(patch2im(Y(:,(r+1):(r+l)), patchsize)))
    title('Actual')
    drawnow
    pause(0.3)
end
%% See what is reconstucted if 1 of the dictionary elements is used only
figure(2)
clf
active_f = 1;
tempB = zeros(K1,N);
tempB(active_f, :) = PI1(active_f, :);
Y_approx = muD1*(muS1.*tempB);
l = (reduceTo - patchsize + 1)^2;
for r = 0:l:2500
    subplot(1, 2, 1)
    recon = histeq(patch2im(Y_approx(:,(r+1):(r+l)), patchsize));
    recon(recon<=0) = 0;
    recon(recon>=1) = 1;
    imshow(recon)
    title('Recon')
    subplot(1, 2, 2)
    imshow(histeq(patch2im(Y(:,(r+1):(r+l)), patchsize)))
    title('Actual')
    drawnow
    disp('blah')
    pause(1)
end
%% Choose features
figure(3)
fs = [];
for k = 1:K1
    if entropy(PI1(k, :)) >= 0.1
        fprintf('F: %d, Hx: %f\n', k, entropy(PI1(k, :)))
        fs = [fs k];
    end
end
imagesc(PI1), colorbar
%% Plot all features

% muD1_new = normalize(muD1);
muD1_new = muD1;
figure(2)
clf
gridsize = 5;
l = (reduceTo - patchsize + 1)^2;
subplot(gridsize, gridsize, 1)
imshow(normalize(patch2im(Y(:,1:l), patchsize)))
sb = 2;
active_f = 0;
for sb = 2:2:25
%     active_f = fs(sb - 1);
    active_f = active_f + 1;
    tempB = zeros(K1,N);
    tempB(active_f, :) = 1;
    Y_approx = muD1_new*(muS1.*tempB);
    
    subplot(gridsize, gridsize, sb)
    recon = normalize(patch2im(Y_approx(:,1:l), patchsize));
    imshow(recon)
    title(sprintf('Feature %d', active_f))
    
    subplot(gridsize, gridsize, sb + 1)
    recon = normalize(patch2im(Y_approx(:,(l + 1):2*l), patchsize));
    imshow(recon)
    title(sprintf('Feature %d', active_f))
    
end
figure(4)
Y_approx = muD1_new*(muS1.*PI1);
recon = normalize(patch2im(Y_approx(:,1:l), patchsize));
imshow(recon)  
%% Plot Higher level features
figure(2)
clf
active_f2 = 8;
b2 = binornd(ones(K2, N), PI2);
y2_approx = muD2*(muS2.*b2);
pi1_transfer = 1./(1 + exp(y2_approx(:, active_f2)));

gridsize = 5;
l = (reduceTo - patchsize + 1)^2;
subplot(gridsize, gridsize, 1)
imshow(histeq(patch2im(Y(:,1:l), patchsize)))
sb = 2;
for sb = 2:25
    tempB = repmat(binornd(ones(K1, 1), pi1_transfer), 1, N);
    Y_approx = muD1*(muS1.*tempB);
    
    subplot(gridsize, gridsize, sb)
    recon = histeq(patch2im(Y_approx(:,1:l), patchsize));
    imshow(recon)  
    
end
figure(4)
Y_approx = muD1*(muS1.*PI1);
recon = histeq(patch2im(Y_approx(:,1:l), patchsize));
imshow(recon)  









