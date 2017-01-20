clear
close all
load('s01.mat')
%%
p1_data = data;
p1_labels = labels;


%%
clf
zx1 = 1;
zx2 = 4;
scatter(labels(:,zx1), labels(:,zx2))
a = 1:size(labels,1);
e = labels(:,zx1) + 0.1;
f = labels(:,zx2) + 0.1;
b = num2str(a');
text(e, f, cellstr(b));
%%
points = labels';
m = size(points,2);
dm = zeros(m,m);
for i = 1:m
    for j = 1:m
        dm(i,j) = norm(points(:,i) - points(:,j));
    end
end
imagesc(dm)
%%
clc
n_movies = 3;
movies = [1 2 29];
% take movies 1 8 30 32
de = 32; % Number of channels
n_samples = size(p1_data, 3); % Num. of samples per movie

first_n_samples = 200;

Y = zeros(de, floor(2*n_movies*first_n_samples)); % Downsample by factor 5
currentIndex = 1;
for i = 1:n_movies
    m = movies(i);
%     s = (i - 1)*first_n_samples + 1;
%     f = i*first_n_samples;
    Y(:, currentIndex:(currentIndex + first_n_samples - 1)) = squeeze(p1_data(m, 1:32, 1:first_n_samples));
    currentIndex = currentIndex + first_n_samples;
end

% EXTRA SAMPLES

for i = 1:n_movies
    m = movies(i);
%     s = (i - 1)*first_n_samples + 1;
%     f = i*first_n_samples;
    Y(:, currentIndex:(currentIndex + first_n_samples - 1)) = squeeze(p1_data(m, 1:32, first_n_samples:(2*first_n_samples - 1)));
    currentIndex = currentIndex + first_n_samples;
end
%%
plot(squeeze(p1_data(1,1,1:600)), 'b')
%%
% clear p1_data p1_labels
% clear data labels
%% INITIALIZE ESSENTIALS
frobNorm = @(M) sum(diag(M'*M));
K = 60;

n_iters = 400; % Atleast 10 (burn-in length as of now)
N = size(Y, 2);
D = zeros(de, K, 2);
S = zeros(K, N, 2); % Need to load-save simultaneously

gm_d = zeros(de, K, 2);
alpha_d = 0.25;
beta_d = 0.000003;

gm_s = zeros(K, N, 2);
alpha_s = 0.6;
beta_s = 0.0003;

gm_n = zeros(de, N, 2);
alpha_n = 1; % 10
beta_n = 1;
%% SAMPLE THE INITIAL VALUES

gm_d(:,:,1) = gamrnd(alpha_d, 1/beta_d, de, K);
gm_s(:,:,1) = gamrnd(alpha_s, 1/beta_s, K, N);
gm_n(:,:,1) = gamrnd(alpha_n, 1/beta_n, de, N);

D(:, :, 1) = normrnd(zeros(de,K), 1./sqrt(gm_d(:,:,1)), [de, K]);
S(:, :, 1) = normrnd(zeros(K,N), 1./sqrt(gm_s(:,:,1)), [K, N]);

%% GIBBS SAMPLING
for iter2 = 1:n_iters
    iter = 2; % Fake iter
    iold = iter - 1;
    % Copy the results from this iteration into iter-1
    % Save all matrices and hyperparameters
    
    tic % START MEASURING TIME
    
    for u = 1:de
        for v = 1:K
            % Update D
            t1 = ...
                sum(gm_n(u,:,iold).*S(v, :, iold).*(Y(u, :) - ...
                D(u, [1:(v-1) (v+1):K], iold)*S([1:(v-1) (v+1):K],:,iold)));
            t2 = gm_d(u, v, iold) + sum(gm_n(u,:,iold).*(S(v,:,iold).^2));
            M = t1/t2;
            GM = t2;
            D(u, v, iter) = normrnd(M, 1/sqrt(GM));
            
            ALPHA = alpha_d + 1/2;
            BETA = beta_d + 0.5*D(u, v, iter)^2;
            gm_d(u, v, iter) = gamrnd(ALPHA, 1/(BETA));
        end
    end
    
    for p = 1:K
        for q = 1:N
            % Update S
            t1 = ...
                sum(gm_n(:,q,iold).*D(:,p,iter).*(Y(:, q) - ...
                D(:, [1:(p-1) (p+1):K], iter)*S([1:(p-1) (p+1):K],q,iold)));
            t2 = gm_s(p, q, iold) + sum(gm_n(:,q,iold).*(D(:,p,iter).^2));
            M = t1/t2;
            GM = t2;
            S(p, q, iter) = normrnd(M, 1/sqrt(GM));
            
            ALPHA = alpha_s + 1/2;
            BETA = beta_s + 0.5*S(p, q, iter)^2;
            gm_s(p, q, iter) = gamrnd(ALPHA, 1/(BETA));
        end
    end
    
    for i = 1:de
        for j = 1:N
            % Update gm_n
            ALPHA = alpha_n + 1/2;
            BETA = beta_n + 0.5*(Y(i,j) - D(i,:,iter)*S(:,j,iter)).^2;
            gm_n(i, j, iter) = gamrnd(ALPHA, 1/(BETA));
        end
    end
    
    toc % REPORT EXECUTION TIME
    % Copy outputs back
    D(:,:,iter - 1) = D(:,:,iter);
    S(:,:,iter - 1) = S(:,:,iter);
    gm_n(:, :, iter - 1) = gm_n(:, :, iter);
    gm_d(:, :, iter - 1) = gm_d(:, :, iter);
    gm_s(:, :, iter - 1) = gm_s(:, :, iter);
    
    % Saved the outputs to files
    filename = ['my_gibbs_matrices/iter_' num2str(iter2) '.mat'];
    IterResult = struct();
    IterResult.tempD = D(:,:,iter);
    IterResult.tempS = S(:,:,iter);
    IterResult.gmd = gm_d(:,:,iter);
    IterResult.gmn = gm_n(:,:,iter);
    IterResult.gms = gm_s(:,:,iter);
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
for c = 1:5
    clf
    plot(Y(:,c), 'r')
    hold on
    plot(3*Y_approx(:,c), 'b');
    hold off
    drawnow
%     figname = sprintf('output_images/fig_recovered_ch%d_%.3f_%.3f,%.3f_%.3f,%.3f_%.3f.png', c, alpha_d, beta_d, alpha_s, beta_s, alpha_n, beta_n);
%     print(h, figname, '-dpng');
    pause(0.3)
end
% clear Y_approx

%%
figure(1)
clf
subplot(2,2,1)
stem(meanD(:,1))
subplot(2,2,2)
stem(meanD(:,40))
subplot(2,2,3)
stem(meanD(:,42))
subplot(2,2,4)
imagesc(meanD)
%%
figure(1)
clf
% plot(meanD(:,1:3))
imagesc(10*meanS)
% plot(meanS(:,1:10))
%%
%%
figure(1)
md = meanD;
mx = max(max(meanD));
mn = min(min(meanD));
md = md./(mx - mn);

subplot(2,2,1)
stem(md(:,1))
subplot(2,2,2)
stem(md(:,40))
subplot(2,2,3)
stem(md(:,42))
subplot(2,2,4)
imagesc(md)
%%
figure(2)
clf
stem(md(:,31))
% axis([0 32 -1 1])
%%
figure(1)
clf
ms = meanS;
mx = max(max(ms));
mn = min(min(ms));
m = mean(mean(ms));
ms = (ms - m)./(mx - mn);
figure(1)
clf
j = 0;

usedElems = cell(6,1);

for i = 1:first_n_samples:N/2
    j = j+1;
    subplot(3,2,j)
%     size(median(ms(:,i:(i + N/n_movies - 2)), 2))
    sm = sum(ms(:,i:(i + first_n_samples - 1)).^2, 2);
    [B,I] = sort(sm);
    usedElems{j} = I(1:5);
    disp('----------------------------------------')
    disp(['Movie ' num2str(movies(j))])
    disp(I(1:5)')
    stem(sm)
    
    title(['Movie ' num2str(movies(j)) '~~'...
        strjoin(cellstr(num2str(labels(movies(j),:)'))') ])
end

for i = (N/2 + 1):first_n_samples:N
    j = j+1;
    subplot(3,2,j)
%     size(median(ms(:,i:(i + N/n_movies - 2)), 2))
    sm = sum(ms(:,i:(i + first_n_samples - 1)).^2, 2);
    [B,I] = sort(sm);
    usedElems{j} = I(1:5);
    disp('----------------------------------------')
    disp(['Movie ' num2str(movies(j - 3))])
    disp(I(1:5)')
    stem(sm)
    
    title(['Movie ' num2str(movies(j - 3)) '~~'...
        strjoin(cellstr(num2str(labels(movies(j - 3),:)'))') ])
end

%%
% size(ms)
% first_n_samples
figure(1)
clf
s = 17
mean(ms(s, 1:200))
hist(ms(s, 1:200), 60)
%%
figure(2)
clf
cs = usedElems{1};
stem(abs(sum(md(:,cs), 2)), 'r')
hold on
cs = usedElems{4};
stem(abs(sum(md(:,cs), 2)), 'b')

stem(abs(sum(md(:,usedElems{4}), 2)) + abs(sum(md(:,usedElems{1}), 2)), 'g')
hold off

%%
ZZZ = [];
clf
for i = 1:6
    ZZZ(:, i) = sum(md(:, usedElems{i}).^2, 2);
    subplot(2,3, i)
    stem(ZZZ(:,i))
end


