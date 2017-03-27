function [X, S, B, c] = InitAll (YH, YL, K, PI, Gamma)

%% Create variables
c = {};
MH = size(YH, 1);
ML = size(YL, 1);
N = size(YH, 2);
c.MH = MH;
c.ML = ML;
c.N = N; % Only N needs to be updated
c.K = K;

X = zeros(c.MH, c.N);
S = zeros(K, N);
B = zeros(K, N);

%% Sampling of S
zmu = zeros(K, 1);
Is = eye(K);
gms = Gamma.s;
parfor i = 1:N
    S(:, i) = mvnrnd(zmu, (1/gms).*Is);
end

%% Sampling of PI and B (or Z)
parfor k = 1:K %% VERIFIED - parfor is BETTER
    for i = 1:N
        B(k, i) = binornd(1, PI(k));
    end
end

fprintf('Initial Samples Ready...\n')

end