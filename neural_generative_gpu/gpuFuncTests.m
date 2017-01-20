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

%%
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