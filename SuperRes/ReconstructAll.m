function [ YH_approx, YL_approx, YHL ] = ...
    ReconstructAll( DH, DL, S, B, biasH, biasL, c, YL, locate, onlyYHL )
%GETYHL Summary of this function goes here
%   YL is required for forming YHL
if ~onlyYHL            
    YH_approx = DH*(S.*B) + repmat(biasH, 1, c.N);
    YL_approx = DL*(S.*B) + repmat(biasL, 1, c.N);

    YHL = YH_approx;
    YHL(locate, :) = YL;            
else
    % Just don't create YL_approx
    YH_approx = DH*(S.*B) + repmat(biasH, 1, c.N);

    YHL = YH_approx;
    YHL(locate, :) = YL;
    YL_approx = 0;
end
t = toc;
end

