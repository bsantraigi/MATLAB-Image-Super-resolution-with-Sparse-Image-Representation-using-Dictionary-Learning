clear
clc
iters = 100000;
x_samples = zeros(iters, 1);
gm_samples = zeros(iters, 1);

% alpha_d = 0.625000;
% beta_d = 0.001250;

alpha_d = 2.6;
beta_d = 0.9;

x_samples(1) = normrnd(0,1);
gm_samples(1) = gamrnd(alpha_d, 1/beta_d);
for j=2:iters
    gm_samples(j) = gamrnd(alpha_d + 0.5, 1/(beta_d + 0.5*x_samples(j-1)^2));
    x_samples(j) = normrnd(0, sqrt(1/gm_samples(j)));
    
%     gm_samples(j) = gamrnd(alpha_d, 1/beta_d);
%     x_samples(j) = normrnd(0, sqrt(1/gm_samples(j)));
end
figure(2)
clf
h = hist(x_samples(2000:size(x_samples, 1)), 100);
hist(x_samples(2000:size(x_samples, 1)), 100)
m = mean(x_samples(2000:size(x_samples, 1)))
v = var(x_samples(2000:size(x_samples, 1)))
hold on
z = normrnd(m, sqrt(v), 18000, 1);
% hist(z)
% axis([-5 5 0 max(h)*1.1])

%%
c = 1.5; % Decrease to increase variance
k = 0.5; % Increase to increase mean
alpha_d = c*k^2;
beta_d = 0.001*c*k;
fprintf('HI\n\nalpha_d = %f;\nbeta_d = %f;\n', alpha_d, beta_d)
z = gamrnd(alpha_d, 1/beta_d, 100000, 1);
fprintf('\nmean = %f;\nvar = %f;\n\n', mean(z), var(z))
fprintf('\nmean = %f;\nvar = %f;\n\n', alpha_d/beta_d, alpha_d/beta_d^2)
figure(2)
clf
hist(z, 100)
%%
c = 10; % Decrease to increase variance
k = 0.2; % Increase to increase mean
alpha_d = c*k^2;
beta_d = 0.0001*c*k;
fprintf('HI\n\nalpha_d = %f;\nbeta_d = %f;\n', alpha_d, beta_d)
z = gamrnd(alpha_d, 1/beta_d, 100000, 1);
fprintf('\nmean = %f;\nvar = %f;\n\n', mean(z), var(z))
fprintf('\nmean = %f;\nvar = %f;\n\n', alpha_d/beta_d, alpha_d/beta_d^2)
figure(2)
clf
hist(z, 100)
%%
c = 0.07; % Decrease to increase variance
k = 10; % Increase to increase mean
alpha_d = c*k^2;
beta_d = c*k;
fprintf('HI\n\nalpha_d = %0.10f;\nbeta_d = %0.10f;\n', alpha_d, beta_d)
z = gamrnd(alpha_d, 1/beta_d, 1000000, 1);
fprintf('\nmean = %f;\nvar = %f;\n\n', mean(z), var(z))
fprintf('\nmean = %f;\nvar = %f;\n\n', alpha_d/beta_d, alpha_d/beta_d^2)
figure(2)
clf
hist(z, 100)

%% Normal Gamma
N = 10000;
samples = zeros(N,1);
alphas = [1, 1, 1, 0.1, 0.01, 0.1];
betas = [1, 10, 1000, 10, 1, 1];
for f = 1:length(alphas)
    alpha = alphas(f);
    beta = betas(f);
    for i =1:N
        g = gamrnd(alpha, 1/beta);
        samples(i) = normrnd(0, 1/sqrt(g));
    end
    subplot(3,2,f);
    h = histc(samples, -100:2:100);
%     plot(-100:2:100, h)
    h2 = interp1(-100:2:100, h, -100:2:100, 'spline');
    plot(-100:2:100, h2)
    title(['\alpha =' sprintf('%0.2f, ', alpha) '\beta =' sprintf('%0.2f', beta)])
end