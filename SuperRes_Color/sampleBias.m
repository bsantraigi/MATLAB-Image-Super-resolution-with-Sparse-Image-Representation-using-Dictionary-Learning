function [bias] = sampleBias(Y, D, S, B, Gamma, c, flag)
    if flag == 'H'
        gam_bias = Gamma.biasH;
        gam_n = Gamma.nH;
        M = c.MH;
    else
        gam_bias = Gamma.biasL;
        gam_n = Gamma.nL;
        M = c.ML;
    end
    G = c.N * gam_n + gam_bias;
    mu = (1/G).*sum(Y - D*(S.*B), 2)*gam_n;
    bias = mvnrnd(mu, (1/G)*eye(M))';
end