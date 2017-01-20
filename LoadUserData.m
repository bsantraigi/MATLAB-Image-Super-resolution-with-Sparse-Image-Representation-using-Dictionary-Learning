function [ Y ] = LoadUserData( user_id, C, T, N )
%LOADUSERDATA Summary of this function goes here
%   Detailed explanation goes here

f = sprintf('s%02d.mat', user_id);
disp(['Loaded data from ' f])
load(f);

Y = zeros(C,T,N);

n_movies = N;
lastM = 0;
t = 0;
for p = 1:N
    m = ceil(p/(N/n_movies));
%     fprintf('Movie Loaded %d\n', m);
    if lastM == m
        % increment in time axis
        t = t + T;
        Y(:,:,p) = data(m, 1:C, (t + 1):(t + T));        
    else
        t = 0;
        Y(:,:,p) = data(m, 1:C, 1:T);
    end
    lastM = m;
end

end

