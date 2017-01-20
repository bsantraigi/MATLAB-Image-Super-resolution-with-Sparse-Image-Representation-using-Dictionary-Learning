function [bias] = sampleBias(Y, D, S, B, Gamma, c)

    G = c.N * Gamma.n + Gamma.bias;
    mu = (1/G).*sum(Y - D*(S.*B), 2)*Gamma.n;
    bias = mvnrnd(mu, (1/G)*eye(c.M))';
end