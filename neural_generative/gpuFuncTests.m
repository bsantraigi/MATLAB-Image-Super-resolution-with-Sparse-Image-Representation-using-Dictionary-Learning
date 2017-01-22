tic
h_e = zeros(64, 1);
h_e = mvnrnd(zeros(64, 1), eye(64));
t = toc;
fprintf('CPU time: %f ms\n', t*1000);

tic
d_e = gpuArray(h_e);
d_e(:) = mvnrnd(zeros(64, 1), eye(64));
t = toc;
fprintf('GPU time: %f ms\n', t*1000);

%% Nested for loop with outer parfor
tic
K = 100;
N = 6000;
B = zeros(K, N);
for k = 1:K
    pi = betarnd(10, 1);
    for n = 1:N
        B(k, n) = binornd(1, pi);
    end
end
t = toc;
fprintf('CPU time: %f ms\n', t*1000);

%%
tic
K = 100;
N = 6000;
B = zeros(K, N);
parfor k = 1:K
    pi = betarnd(10, 1);
    for n = 1:N
        B(k, n) = binornd(1, pi);
    end
end
t = toc;
fprintf('Parallel CPU time: %f ms\n', t*1000);

%%

K = 100;
pi = zeros(K, 1);
tic
for k = 1:K
    pi(k) = betarnd(10, 1);
end
t = toc;
fprintf('CPU time: %f ms\n', t*1000);

tic
K = 100;
pi = gpuArray(pi);
tic
pi2 = gather(pi);
for k = 1:K
    pi2(k) = betarnd(10, 1);
end
pi = gpuArray(pi2);
t = toc;
fprintf('GPU time: %f ms\n', t*1000);

%% BLAS

h_N = 2000;
h_a = rand(h_N, h_N);
h_y = rand(h_N, h_N);
tic
h_b = h_a + h_a*h_y + (h_a - h_y)*h_y';
t = toc;
fprintf('CPU time: %f ms\n', t*1000);


d_a = gpuArray(h_a);
d_y = gpuArray(h_y);
tic
d_b = d_a + d_a*d_y + (d_a - d_y)*d_y';
t = toc;
fprintf('GPU time: %f ms\n', t*1000);

e = h_b - d_b;
e(1:10,1:10)

clear

%% BLAS 2
h_N = 2000;
M = 1000;
h_a = rand(h_N, h_N);
h_y = rand(h_N, h_N);
tic
h_b = h_a(:,1:M) + h_a*h_y(:,1:M) + (h_a - h_y)*h_y(1:M, :)';
t = toc;
fprintf('CPU time: %f ms\n', t*1000);


d_a = gpuArray(h_a);
d_y = gpuArray(h_y);
tic
d_b = d_a(:,1:M) + d_a*d_y(:,1:M) + (d_a - d_y)*d_y(1:M, :)';
t = toc;
fprintf('GPU time: %f ms\n', t*1000);

e = h_b - d_b;
e(1:10,1:10)

clear

%% SVD on gpu
clear
k = 1000;
A = rand(k, k) + eye(k);
tic
[~, ~, ~] = svd(A);
t = toc;
fprintf('CPU time: %f ms\n', t*1000);


d_A = gpuArray(A);
tic
[~, ~, ~] = svd(d_A);
t = toc;
fprintf('GPU time: %f ms\n', t*1000);

%% Arrayfun on GPU

M = randi(10, 4, 4)
N = cell(4, 1);
for i = 1:4
    N{i} = randi(10, 4, 1);
end

f = @(v) TestUnitF(M, v);
out = arrayfun(f, N, 'UniformOutput', false)

%% Sigmoid inverse
tic
Y2 = sigmoid_Inv(post_PI);
toc

