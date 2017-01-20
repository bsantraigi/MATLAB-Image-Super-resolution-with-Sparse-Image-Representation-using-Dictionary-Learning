function [D, S, gm_d, gm_s, gm_n] = ElemGibbsSampleNext...
    (Y, D, S, gm_d, gm_s, gm_n, constants, updateD )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    [de, K] = size(D);
    N = size(S,2);
    alpha_d = constants.alpha_d;
    beta_d = constants.beta_d;
    alpha_s = constants.alpha_s;
    beta_s = constants.beta_s;
    alpha_n = constants.alpha_n;
    beta_n = constants.beta_n;
    
    if updateD
        for u = 1:de
            for v = 1:K
                % Update D
                t1 = ...
                    gm_n*sum(S(v, :).*(Y(u, :) - ...
                    D(u, [1:(v-1) (v+1):K])*S([1:(v-1) (v+1):K],:)));
                t2 = gm_d(u, v) + gm_n*sum((S(v,:).^2));
                M = t1/t2;
                GM = t2;
                D(u, v) = normrnd(M, 1/sqrt(GM));

                ALPHA = alpha_d + 1/2;
                BETA = beta_d + 0.5*D(u, v)^2;
                gm_d(u, v) = gamrnd(ALPHA, 1/(BETA));
            end
        end
    end
    
    for p = 1:K
        for q = 1:N
            % Update S
            t1 = ...
                gm_n*sum(D(:,p).*(Y(:, q) - ...
                D(:, [1:(p-1) (p+1):K])*S([1:(p-1) (p+1):K],q)));
            t2 = gm_s(p, q) + gm_n*sum((D(:,p).^2));
            M = t1/t2;
            GM = t2;
            S(p, q) = normrnd(M, 1/sqrt(GM));
            
            ALPHA = alpha_s + 1/2;
            BETA = beta_s + 0.5*S(p, q)^2;
            gm_s(p, q) = gamrnd(ALPHA, 1/(BETA));
        end
    end
    
    % Update gm_n
    ALPHA = de*N*(alpha_n - 0.5) + 1;
    BETA = beta_n*de*N + 0.5*norm(Y - D*S, 'fro').^2;
    gm_n = gamrnd(ALPHA, 1/BETA);

end

