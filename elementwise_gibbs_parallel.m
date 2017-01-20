clear
close all
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
% dat = squeeze(p1_data(1,1:32,:));
% [~,s,~] = svd(dat);
% plot(diag(s))
% title('Singular values of data matrix of Mov1; EEG 32C; User 1')
% %%
% dat = squeeze(p1_data(29,1:32,:));
% [~,s,~] = svd(dat);
% plot(diag(s))
% title('Singular values of data matrix of Mov29; EEG 32C; User 1')
%%
nice_plot(p1_data, 'Y', 0)
%%
clf
zx1 = 1;
zx2 = 4;
scatter(p1_labels(:,zx1), p1_labels(:,zx2))
a = 1:size(p1_labels,1);
e = p1_labels(:,zx1) + 0.1;
f = p1_labels(:,zx2) + 0.1;
b = num2str(a');
text(e, f, cellstr(b));
%%
Plot_DistanceMat(p1_labels', 1);
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
    Y(:, currentIndex:(currentIndex + first_n_samples - 1)) = squeeze(p1_data(m, 1:32, 1:first_n_samples));
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
K = 60;
N = size(Y, 2);
D = zeros(de, K);
S = zeros(K, N); % Need to load-save simultaneously

gm_d = zeros(de, K);
alpha_d = 0.375000;
beta_d = 0.000750;

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

D(:, :) = normrnd(zeros(de,K), 1./sqrt(gm_d(:,:)), [de, K]);
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
    [D, S, gm_d, gm_s, gm_n] = ElemGibbsSampleNext(Y, D, S, gm_d, gm_s, gm_n, constants, true);
    
    toc % REPORT EXECUTION TIME
    disp(['Burn-in progress: ' num2str(iter2)])
end

for iter2 = 1:n_iters
%     IterResult = {struct(), struct(), struct(), struct()};
    tic % START MEASURING TIME
    spmd
        % call here
        IterResult = struct();
        [IterResult.tempD, IterResult.tempS,...
            IterResult.gmd, IterResult.gms, IterResult.gmn]...
            = ElemGibbsSampleNext(Y, D, S, gm_d, gm_s, gm_n, constants, true);    
    end
    toc % REPORT EXECUTION TIME
    
    % Update [D, S, gm_d, gm_s, gm_n]
    ti = IterResult{1};
    D = ti.tempD;
    S = ti.tempS;
    gm_d = ti.gmd;
    gm_s = ti.gms;
    gm_n = ti.gmn;
    clear ti
    
    % Saved the outputs to files
    for lab = 1:8
        filename = ['my_gibbs_matrices/iter_' num2str(lab) '_' num2str(iter2) '.mat'];
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
meanD = zeros(de, K);
meanS = zeros(K, N);
for iter2 = 1:n_iters
    for lab = 1:8
        filename = ['my_gibbs_matrices/iter_' num2str(lab) '_' num2str(iter2) '.mat'];
        disp(['Read ' filename])
        load(filename)
        meanD = meanD + tempD;
        meanS = meanS + tempS;
        clear tempD tempS gmd gmn gms
    end
end
meanD = meanD/(n_iters*8);
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
nice_plot(meanD, 'D', 15)
%%
nice_plot(meanS, 'S', 15)
%% Top 5 dictionary atoms for each movie - Energywise, SORT BY OCCURENCE
clc
figure(1)
clf
for m = 1:1
    lister = {0};
    top_n = 6;
    offset = (m - 1)*first_n_samples;
    for i = 1:6
        x = i + offset;
        compMat = meanD*diag(meanS(:,x));
        eng = sum(compMat.^2, 1);
        [~, I] = sort(eng, 'descend');
        accept = I(1:top_n);
        lister{i} = accept;
        
        disp(num2str(accept))

        subplot(2, 3, i)
%         plot(eng)
        hold on
        plot(Y(:,x), 'b')
        plot(Y_approx(:,x), 'green')
        plot(sum(compMat(:, accept), 2), 'r')
        legend('Actual Data', 'Reconstructed', ['Using top ' num2str(top_n)])
        hold off
    end
%     lister = vertcat(lister{:});
%     lister = lister(:);
%     atoms = unique(lister);
%     times = hist(lister, atoms);
%     [B, I] = sort(times, 'descend');
%     disp(['Movie: ' num2str(m)])
%     disp(['Atoms: ' num2str(atoms(I)')])
%     disp(['Times: ' num2str(times(I))])
%     modTimes = zeros(60, 1);
%     modTimes(atoms) = times;
%     subplot(10, 4, m);
%     bar(1:60, modTimes)
%     set(gca,'XTickLabel','')
%     set(gca,'YTickLabel','')
%     handle = title(['Mov', num2str(m) '->']);
% 	set(handle,'Position',[-10, 5, 0]);
%     axis([0 60 0 (first_n_samples + 5)])
end
%% Actual Energy of each dictionary atoms present in Approx. - SORT BY ENERGY
normalize = @(v) v./sum(v);
clc
figure(1)
clf
fullEng = zeros(60, 40);
lister = cell(40, first_n_samples);
listerC = cell(40);
for m = 1:40
    offset = (m - 1)*first_n_samples;
    for i = 1:first_n_samples
        x = i + offset;
        compMat = meanD*diag(meanS(:,x));
        eng = sqrt(sum(compMat.^2, 1));
        fullEng(:,m) = fullEng(:, m) + eng';
        [~, I] = sort(eng, 'descend');
        accept = I(1:top_n);
        lister{m, i} = accept;
%         disp(num2str(accept))

%         subplot(2, 3, i)
%         hold on
%         plot(Y(:,x), 'r')
%         plot(Y_approx(:,x), 'green')
%         plot(sum(compMat(:, accept), 2), 'b')
%         legend('Actual Data', 'Reconstructed', ['Using top ' num2str(top_n)])
%         hold off
    end
%     fullEng(:, m) = normalize(fullEng(:, m));
    subplot(5, 8, m)
    plot(fullEng(:,m), 'r'), grid on
%     axis([0 60 0 1])
    
    [~, I] = sort(squeeze(fullEng(:, m)), 'descend');
    accept = I(1:top_n);
    listerC{m} = accept;
end
%%
Plot_DistanceMat(p1_labels', 1);
%%
Plot_DistanceMat(fullEng, 2);
%%
badMat = zeros(32, 40);
for m = 1:40
    badMat(:, m) = meanD*fullEng(:, m);
end
Plot_DistanceMat(badMat, 3);
%%
figure(1)
clf
offset = 0;
for m = 1:5
    atoms = lister{m,1};
%     atoms = listerC{m};
    for c = 1:top_n
        subplot(5, 8, c + offset)
        stem(meanD(:, atoms(c))), grid on
        title(['MOV #' num2str(m) ' ATOM #' num2str(atoms(c))])
    end
    offset = offset + 8;
%     disp(['Movie: ' num2str(m)])
%     disp(['Atoms: ' num2str(atoms(1:5))])
    pause
end
%% STAGE 2 - FORM SUPER DICTIONARY
superd = zeros(32, top_n*40);
currentCol = 1;
for m = 1:40
    disp(['Movie' num2str(m)])
    disp(num2str(listerC{m}'))
    atoms = listerC{m};
    for i = 1:top_n
        superd(:, currentCol) = meanD(:, atoms(i));
        currentCol = currentCol + 1;
    end
end
save('fixedDictionary_0.375_0.00075_train40.mat', 'superd')
%%
zzz = vertcat(lister{:});
% zzz = zzz(:);
%%
superd = meanD;
save('fixedDictionary_0.375_0.00075_train40.mat', 'superd')