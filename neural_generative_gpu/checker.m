%% Beta Distribution
t = 0:0.01:1;
figure(3)

plot(t, betapdf(t, 2, 30))

%%
figure(1)
clf
recon = reshape(S(17, 1)*D(:, 17), patchsize, patchsize);

% recon(recon<=0) = 0;
% recon(recon>=1) = 1;
% recon = (recon - mn)/(mx - mn);
imshow(recon)


%%
figure(1)
for j = 1:100
f = 30;
Vfi = D2(:, f).*S2(f, :); % Forcably activate a level 2 feature
pi_prime = 1./(1 + exp(-Vfi));
B1_back = binornd(ones(length(pi_prime), 1), pi_prime);
I_prime = D*(S(:, i).*B1_back);
% I_prime = D*(S(:, i).*pi_prime);
% I_prime = D*(S(:, i).*B(:, i));

recon = reshape(I_prime, patchsize, patchsize);
recon(recon<=0) = 0;
recon(recon>=1) = 1;
subplot(10,10,j)
imshow(recon)
end
%%
figure(1)
clf
f = 35;
Bx = zeros(K1, size(B, 2));
Bx(f, :) = 1;
imshow(patch2im(D*(S.*Bx), patchsize))




