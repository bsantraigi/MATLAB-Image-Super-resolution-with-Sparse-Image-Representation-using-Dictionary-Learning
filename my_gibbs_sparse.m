clear
close all
load('s01.mat')
%%
p1_data = data;
p1_labels = labels;


%%

clc
n_movies = 4;
de = 32; % Number of channels
n_samples = size(p1_data, 3); % Num. of samples per movie

downsample = 32;

Y = zeros(de, floor(n_movies*n_samples/downsample)); % Downsample by factor 5
for i = 1:n_movies
    s = (i - 1)*n_samples/downsample + 1;
    f = i*n_samples/downsample;
    Y(:, s:f) = squeeze(p1_data(i, 1:32, 1:downsample:8064));
end
%%
clear p1_data p1_labels
clear data labels
%%
frobNorm = @(M) sum(diag(M'*M));
K = 200;

n_iters = 500; % Atleast 10 (burn-in length as of now)
N = size(Y, 2);
D = zeros(de, K, n_iters);
S = zeros(K, N, n_iters); % Need to load-save simultaneously

gm_d = zeros(n_iters, 1);
alpha_d = 1.1421;
beta_d = 5.7120e-05;

gm_s = zeros(n_iters, 1);
alpha_s = 1 % 2
beta_s = 1 % 0.1

gm_n = zeros(n_iters, 1);
alpha_n = 1; % 10
beta_n = 1;
% SAMPLE THE INITIAL VALUES

gm_d(1) = gamrnd(alpha_d, 1/beta_d);
gm_s(1) = gamrnd(alpha_s, 1/beta_s);
gm_n(1) = gamrnd(alpha_n, 1/beta_n);

D(:, :, 1) = normrnd(0, 1/sqrt(gm_d(1)), [de, K]);
S(:, :, 1) = normrnd(0, 1/sqrt(gm_d(1)), [K, N]);

for iter2 = 1:n_iters
    iter = 2; % Fake iter
    % Copy the results from this iteration into iter-1
    % Save all matrices and hyperparameters
    
    tic % START MEASURING TIME
    Q = S(:,:,iter - 1)*S(:,:,iter - 1)' + ...
        gm_d(iter - 1)/gm_n(iter - 1)*eye(K);
    D(:,:,iter) = ...
        matnormrnd(Y*S(:,:,iter - 1)'/Q, eye(de), inv(Q)/gm_n(iter - 1));
    Q = D(:,:,iter)'*D(:,:,iter) + gm_s(iter - 1)/gm_n(iter - 1)*eye(K);
    
    S(:,:,iter) = ...
        matnormrnd(Q\D(:,:,iter)'*Y, inv(Q)/gm_n(iter - 1), eye(N));
    gm_d(iter) = gamrnd(alpha_d + K*de/2, 1/(beta_d + ...
        frobNorm(D(:,:,iter))/2));
    gm_s(iter) = gamrnd(alpha_s + K*N/2, 1/(beta_s + ...
        frobNorm(S(:,:,iter))/2));
    gm_n(iter) = gamrnd(alpha_n + de*N/2, 1/(beta_n + ...
        frobNorm(Y - D(:,:,iter)*S(:,:,iter))/2));
    
    toc % REPORT EXECUTION TIME
    % Copy outputs back
    D(:,:,iter - 1) = D(:,:,iter);
    S(:,:,iter - 1) = S(:,:,iter);
    gm_n(iter - 1) = gm_n(iter);
    gm_d(iter - 1) = gm_d(iter);
    gm_s(iter - 1) = gm_s(iter);    
    
    % Saved the outputs to files
    filename = ['my_gibbs_matrices/iter_' num2str(iter2) '.mat'];
    IterResult = struct();
    IterResult.tempD = D(:,:,iter);
    IterResult.tempS = S(:,:,iter);
    IterResult.gmd = gm_d(iter);
    IterResult.gmn = gm_n(iter);
    IterResult.gms = gm_s(iter);
    save(filename, '-struct', 'IterResult');
    clear IterResult
    disp(['New file ' filename ' saved'])
end

%% READ SAVED FILES AND GET MEAN - WHAT TO DO ABOUT BURN-IN?
meanD = zeros(de, K);
meanS = zeros(K, N);
for iter2 = 200:n_iters
    filename = ['my_gibbs_matrices/iter_' num2str(iter2) '.mat'];
    disp(['Read ' filename])
    load(filename)
    meanD = meanD + tempD;
    meanS = meanS + tempS;
    clear tempD tempS gmd gmn gms
end
meanD = meanD/n_iters;
meanS = meanS/n_iters;
%%
h = figure(1);

Y_approx = meanD*meanS;
for c = 1:40:200
    clf
    plot(Y(:,c), 'r')
    hold on
    plot(Y_approx(:,c), 'b');
    hold off
    drawnow
%     figname = sprintf('output_images/fig_recovered_ch%d_%.3f_%.3f,%.3f_%.3f,%.3f_%.3f.png', c, alpha_d, beta_d, alpha_s, beta_s, alpha_n, beta_n);
%     print(h, figname, '-dpng');
    pause(0.3)
end
clear Y_approx

%%
figure(1)
clf
subplot(2,2,1)
stem(meanD(:,1))
subplot(2,2,2)
stem(meanD(:,50))
subplot(2,2,3)
stem(meanD(:,99))
subplot(2,2,4)
imagesc(meanD)
%%
figure(1)
clf
% plot(meanD(:,1:3))
imagesc(meanS)