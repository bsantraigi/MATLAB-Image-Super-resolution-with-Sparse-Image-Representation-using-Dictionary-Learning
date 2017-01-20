clearvars
close all
C = 32; % First dimension of Y
% Taking 1 sec of data, Fs = 128 Hz, from deap site
T = 128; % Second dimension of Y
N = 60; % Third dimension/depth of Y
Y = MultiUserData([1, 2, 3, 4, 5, 6], C, T, N);

%% Draw fft of data
Fs = 128;
NFFT = 1024; % Next power of 2 from length of y
L = 1024;
Y = squeeze(fft(p1_data(2,1,1:1024),NFFT)/L);
f = Fs/2*linspace(0,1,NFFT/2+1);

% Plot single-sided amplitude spectrum.
figure(1)
clf
plot(f,2*abs(Y(1:NFFT/2+1))) 
title('Single-Sided Amplitude Spectrum of y(t)')
xlabel('Frequency (Hz)')
ylabel('|Y(f)|')

%%
figure(1)
clf
for i = 1:N
    imagesc(Y(:,:,i));
    pause
end
%%
K = 80; % Third dimension/depth of Y
D = zeros(C, T, K);
S = zeros(N, K);
gm_d = zeros(C, 1);
gm_s = zeros(N, K);
gm_n = zeros(C, N);

%% FIRST SAMPLES
% Initialize gm_d
alpha_d = 1.6;
beta_d = 0.07;
for i = 1:C
    gm_d(i) = gamrnd(alpha_d, 1/beta_d);
end

% Initialize gm_s
alpha_s = 1.5;
beta_s = 0.06;
for i = 1:N
    for j = 1:K
        gm_s(i, j) = gamrnd(alpha_s, 1/beta_s);        
    end
end


% Initialize gm_n
alpha_n = 3;
beta_n = 0.3;

for i = 1:C
    for j = 1:N
        gm_n(i, j) = gamrnd(alpha_n, 1/beta_n);
    end
end

constants = struct();
constants.alpha_d = alpha_d;
constants.beta_d = beta_d;
constants.alpha_s = alpha_s;
constants.beta_s = beta_s;
constants.alpha_n = alpha_n;
constants.beta_n = beta_n;

% Initialize D
for j = 1:T
    for i = 1:C
        for k = 1:K
            if j == 1
                D(i, j, k) = normrnd(0, sqrt(1/gm_d(i)));
            else
                D(i, j, k) = normrnd(D(i, j - 1, k), sqrt(1/gm_d(i)));
            end
        end
    end
end

% Initialize S
for p = 1:N
    for k = 1:K
        S(p, k) = normrnd(0, 1/sqrt(gm_s(p, k)));
    end
end

display('Initial Sampling Done')

%%
figure(1)
clf
imagesc(D(:,:,1));
%% START SAMPLING
iters = 600;
burn = 250;
meanD = zeros(C,T,K);
meanS = zeros(N,K);
for it = 1:iters
    disp(['Iteration:' num2str(it)])
    if it > burn
        meanD = meanD + D;
        meanS = meanS + S;
    end
    tic
    [D, S, gm_d, gm_s, gm_n] = GibbsSampleNextTensor_V2(Y, D, S, gm_d, gm_s, gm_n, constants, true);
    toc
end
%%
meanD = meanD/(iters - burn);
meanS = meanS/(iters - burn);
Y_approx = zeros(C, T, N);
for j = 1:T
    Y_approx(:,j,:) = squeeze(meanD(:,j,:))*meanS';
end
save('fullWSpace_MultiUser_1to6_N60_T128');
display('Mean Calculated...')
%%
close all
figure(1)
for c = 1:32
    clf
    hold on
    plot(Y(c, :, 1), 'r')
    plot(Y_approx(c, :, 1), 'b')
    hold off
    legend('Actual', 'Approx');
    pause
end
%% Check if approximation is good
close all
figure(1)
clf

for k = 40:N
    clf
    subplot(2, 1, 1)
    M = squeeze(Y(:,:,k));
    imagesc(M), colorbar
    xlabel('Time n')
    ylabel('Channel i(1~32)')
    title('Actual Data')
    m1 = min(M(:));
    m2 = max(M(:));
    subplot(2, 1, 2)
    imagesc(Y_approx(:,:,k), [m1, m2]), colorbar
    xlabel('Time n')
    ylabel('Channel i(1~32)')
    title('Reconstructed Data')
    pause
end
%%
figure(1)
clf
for k = 1:K
    imagesc(meanD(:,:,k)), colorbar
    pause
end
%%
figure(1)
clf
imagesc(meanS'), colorbar

%%
figure(1)
clf
for k = 1:N
    imagesc(Y(:,:,k)), colorbar
    pause
end