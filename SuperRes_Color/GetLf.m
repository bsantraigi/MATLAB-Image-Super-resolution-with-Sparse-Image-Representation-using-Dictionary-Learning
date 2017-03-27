function [ Lf ] = GetLf( patchsize_hres, patchsize_lres )
%GETLF Summary of this function goes here
%   Detailed explanation goes here
pH = patchsize_hres;
pL = patchsize_lres;
pHsQ = pH^2;
pLsQ = pL^2;

LfB = zeros(pHsQ, pHsQ);

sigma_blur = 1;
es = exp(-1/(2*sigma_blur^2));
for r = 1:pHsQ
    % Each row will have 9 non-zero coefficients
    % for the 8 neighbours and its own pixel location
    if floor((r-1)/pH) == 0 || mod(r, pH) == 0 || ...
            mod(r-1, pH) == 0 || ceil(r/pH) == pH
        continue
    end
    LfB(r, r) = 1;
    LfB(r, [r - 1, r + 1, r - pH, r + pH]) = es;
    LfB(r, [r - pH + 1, r - pH - 1, ...
        r + pH + 1, r + pH - 1]) = es^2;
    LfB(r, :) = LfB(r, :)/sum(LfB(r, :));
end

LfD = zeros(pLsQ, pHsQ);

for r = 1:pLsQ
    rx = 2*(floor((r - 1)/pL) + 1);
    ry = 2*(mod(r - 1, pL) + 1);
    rH = (rx - 1)*pH + ry;
    LfD(r, rH) = 1;
end
Lf = LfD * LfB;

end

