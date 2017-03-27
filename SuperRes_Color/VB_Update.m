function [ muD, muS, PI, Gamma, Palpha, Pbeta ] = VB_Update( Y, muD, muS, PI, Gamma, Palpha, Pbeta, Alpha, Beta, c )
%VB_UPDATE Summary of this function goes here
%   Detailed explanation goes here

% Initially Palpha should be set to Alpha explicitly

M = c.M;
K = c.K;
N = c.N;
% fprintf('N %d,K %d ,M %d', N, K, M)

% Update B
for i = 1:N
    for k = 1:K
        notk = [1:k-1, (k+1):K];
        exp_del_Yik = Y(:,i) - (muD(:,notk)*(muS(notk,i).*PI(notk, i)));
        innerF = (muS(k, i)^2 + Gamma.CovS(k, k, i)) * (muD(:,k)'*muD(:,k) + M/Gamma.d(k)) - 2*muS(k, i).*(muD(:, k)'*exp_del_Yik);
        p1 = psi(Palpha.pi(k)) - psi(Palpha.pi(k) + Pbeta.pi(k)) - ...
            0.5*Palpha.n/Pbeta.n*innerF;

        p0 = psi(Pbeta.pi(k)) - psi(Palpha.pi(k) + Pbeta.pi(k));
        
%         fprintf('p1 split: %f, %f\n', ...
%             (muS(k, i)^2 + Gamma.CovS(k, k, i)), (muD(:,k)'*muD(:,k) + M/Gamma.d(k)));
        p1 = exp(p1);
        p0 = exp(p0);
%         fprintf('p0 %f, p1 %f\n', p0, p1)
        if isinf(p1)
            p1 = 1;
        elseif isinf(p0)
            p1 = 0;
        else
            s = p0 + p1;
            p1 = p1/s;
            p0 = p0/s;
        end
%         fprintf('normed: p0 %f, p1 %f\n', p0, p1)
        if isnan(p1)
            disp('NAN in update B. Code terminated')
            return
        end
        PI(k, i) = p1;
        
    end
end

% Update Post_PI
% for i = 1:N
%     Palpha.pi(i) = Alpha.pi/N + sum(PI(:, i));
%     Pbeta.pi(i) = Beta.pi*(N - 1)/N + K - sum(PI(:, i));
% end

for k = 1:K
    Palpha.pi(k) = Alpha.pi/K + sum(PI(k, :));
    Pbeta.pi(k) = Beta.pi*(K - 1)/K + N - sum(PI(k, :));
end

% Update D
for k = 1:K
    fact_p = 0;
    fact_u = zeros(M,1);
    notk = [1:k-1, (k+1):K];
    for i = 1:N
        fact_p = fact_p + (Gamma.CovS(k,k,i) + muS(k, i)^2)*PI(k, i);        
        exp_del_Yik = Y(:,i) - (muD(:, notk)*(muS(notk,i).*PI(notk, i)));
        fact_u = fact_u + (PI(k, i) * muS(k, i)) .* exp_del_Yik;
    end
    Gamma.d(k) = Palpha.n/Pbeta.n*fact_p + Palpha.d/Pbeta.d;
    muD(:, k) = (1/Gamma.d(k)) * Palpha.n/Pbeta.n * fact_u;
    if any(isnan(fact_u))
        disp('NAN in Update D. Code terminated')
        return
    end
end

% Update S
for i = 1:N
    M1 = muD'*muD + diag(M./Gamma.d);
    M2 = PI(:, i)*PI(:, i)' + diag( PI(:, i) - PI(:, i).^2 );
    ex_DB_T_DB = M1.*M2;

    Gamma.PrecS(:,:,i) = (Palpha.n/Pbeta.n).*ex_DB_T_DB + (Palpha.s/Pbeta.s).*eye(K);
    Gamma.CovS(:,:,i) = inv(Gamma.PrecS(:,:,i));
    ex_DB = muD.*repmat(PI(:, i)', M, 1);
    muS(:,i) = (Palpha.n/Pbeta.n).*(Gamma.CovS(:,:,i)*ex_DB'*Y(:,i));
end

% Update for Gamma.n
Palpha.n = Alpha.n + M*N/2;
fact_n = 0;
for i = 1:N
    e0 = norm(Y(:,i) - muD*(muS(:,i).*PI(:, i)))^2;
    e1 = 0;
    e2 = 0;
    e3 = 0;
    notk = [1:k-1, (k+1):K];
    for k = 1:K
        for l = notk
            e1 = e1 + PI(k, i)*PI(l, i)*Gamma.CovS(k, l, i) .* (muD(:, k)'*muD(:, l));
        end
        e2 = e2 + (muD(:, k)'*muD(:, k) + M/Gamma.d(k))*(muS(k,i)^2 + Gamma.CovS(k, k, i))*PI(k, i);
        e3 = e3 - (PI(k, i)^2*muS(k, i)^2)*(muD(:, k)'*muD(:, k));
    end
    fact_n = fact_n + e0 + e1 + e2 + e3;
end
Pbeta.n = Beta.n + fact_n/2;


% Update for Gamma.s
Palpha.s = Alpha.s + K*N/2;
fact_s = 0;
for i = 1:N
    for k = 1:K
        fact_s = fact_s+ Gamma.CovS(k, k, i);
    end
end
Pbeta.s = Beta.s + 0.5*(norm(muS, 'fro')^2 + fact_s);

% Update for Gamma.d
Palpha.d = Alpha.d + K*M/2;
fact_d = 0;
for k = 1:K
    fact_d = fact_d+ M/Gamma.d(k);
end

Pbeta.d = Beta.d + 0.5*(norm(muD, 'fro')^2 + fact_d);

end

