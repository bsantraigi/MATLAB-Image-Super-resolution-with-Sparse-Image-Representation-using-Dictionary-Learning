clc
clear
close all
load('fixedDictionary_0.375_0.00075_train40.mat')
superD = superd;
clear superd
%%
load('s01.mat')
p1_data = data;
p1_labels = labels;

clear data labels

load('s02.mat')
p1_data2 = data;
p1_labels2 = labels;

clear data labels
%%
clc
close all
n_movies = 40;
% movies = [1 2 29 30 10 21 39 40];
% movies = [1 2];
movies = 1:n_movies;
% take movies 1 8 30 32
de = 32; % Number of channels
n_samples = size(p1_data, 3); % Num. of samples per movie

first_n_samples = 10;

Y = zeros(de, floor(n_movies*first_n_samples)); % Downsample by factor 5
currentIndex = 1;
for i = 1:n_movies
    m = movies(i);
%     s = (i - 1)*first_n_samples + 1;
%     f = i*first_n_samples;
    Y(:, currentIndex:(currentIndex + first_n_samples - 1)) = squeeze(p1_data(m, 1:32, 101:(100 + first_n_samples)));
    currentIndex = currentIndex + first_n_samples;
end

% SAMPLES FROM SECOND PERSON

% for i = 1:n_movies
%     m = movies(i);
% %     s = (i - 1)*first_n_samples + 1;
% %     f = i*first_n_samples;
%     Y(:, currentIndex:(currentIndex + first_n_samples - 1)) = squeeze(p1_data2(m, 1:32, first_n_samples:(2*first_n_samples - 1)));
%     currentIndex = currentIndex + first_n_samples;
% end

%%
% clear p1_data p1_labels

%% INITIALIZE ESSENTIALS
frobNorm = @(M) sum(diag(M'*M));
K = size(superD, 2);
N = size(Y, 2);
S = zeros(K, N); % Need to load-save simultaneously

gm_d = zeros(de, K);
alpha_d = 0.625000;
beta_d = 0.001250;

% alpha_d = 0.6;
% beta_d = 0.0003;

gm_s = zeros(K, N);
alpha_s = 0.6;
beta_s = 0.0003;

% alpha_s = 3;
% beta_s = 0.3;

gm_n = 0;
alpha_n = 3; % 10
beta_n = 0.3;

constants = struct();
constants.alpha_d = alpha_d;
constants.beta_d = beta_d;
constants.alpha_s = alpha_s;
constants.beta_s = beta_s;
constants.alpha_n = alpha_n;
constants.beta_n = beta_n;

%% SAMPLE THE INITIAL VALUES

gm_d(:,:) = gamrnd(alpha_d, 1/beta_d, de, K);
gm_s(:,:) = gamrnd(alpha_s, 1/beta_s, K, N);
gm_n = gamrnd(alpha_n, 1/beta_n);

S(:, :) = normrnd(zeros(K,N), 1./sqrt(gm_s(:,:)), [K, N]);

%% CREATE PARPOOL WITH 4 WORKERS
parpool(8)
%% GIBBS SAMPLING
burn_in = 150;
n_iters = 200;
for iter2 = 1:burn_in
    % Copy the results from this iteration into iter-1
    % Save all matrices and hyperparameters
    
    tic % START MEASURING TIME
    % call here
    [~, S, gm_d, gm_s, gm_n] = ElemGibbsSampleNext(Y, superD, S, gm_d, gm_s, gm_n, constants, false);
    
    toc % REPORT EXECUTION TIME
    disp(['Burn-in progress: ' num2str(iter2)])
end

for iter2 = 1:n_iters
%     IterResult = {struct(), struct(), struct(), struct()};
    tic % START MEASURING TIME
    spmd
        % call here
        IterResult = struct();
        [~, IterResult.tempS,...
            IterResult.gmd, IterResult.gms, IterResult.gmn]...
            = ElemGibbsSampleNext(Y, superD, S, gm_d, gm_s, gm_n, constants, false);    
    end
    toc % REPORT EXECUTION TIME
    
    % Update [D, S, gm_d, gm_s, gm_n]
    ti = IterResult{1};
    S = ti.tempS;
    gm_d = ti.gmd;
    gm_s = ti.gms;
    gm_n = ti.gmn;
    clear ti
    
    % Saved the outputs to files
    for lab = 1:8
        filename = ['wonly_gibbs_matrices/iter_' num2str(lab) '_' num2str(iter2) '.mat'];
        ti = IterResult{lab};
        save(filename, '-struct', 'ti');
        disp(['New file ' filename ' saved'])
        clear ti
    end

    clear IterResult
end
disp('WHOO FINISHED!!!')
%% DELETE PARPOOL
delete(gcp)
%% READ SAVED FILES AND GET MEAN - WHAT TO DO ABOUT BURN-IN?
n_iters = 200;
meanD = superD;
meanS = zeros(K, N);
for iter2 = 1:n_iters
    for lab = 1:8
        filename = ['wonly_gibbs_matrices/iter_' num2str(lab) '_' num2str(iter2) '.mat'];
        disp(['Read ' filename])
        load(filename)
        meanS = meanS + tempS;
        clear tempS gmd gmn gms
    end
end
meanS = meanS/(n_iters*8);
%%
h = figure(1);

Y_approx = meanD*meanS;
for c = 1:6
    subplot(2,3,c)
    plot(Y(:,c), 'r')
    hold on
    plot(Y_approx(:,c), 'b');
    hold off
    drawnow
%     figname = sprintf('output_images/fig_recovered_ch%d_%.3f_%.3f,%.3f_%.3f,%.3f_%.3f.png', c, alpha_d, beta_d, alpha_s, beta_s, alpha_n, beta_n);
%     print(h, figname, '-dpng');
    pause(0.3)
end
% clear Y_approx
%%
nice_plot(meanS, 'S', 0);
%%
nice_plot(superD, 'D', 15);
%%
figure(1)
clf
m = 19;
imagesc(meanS(:, (m*first_n_samples + 1):((m + 1)*first_n_samples)))