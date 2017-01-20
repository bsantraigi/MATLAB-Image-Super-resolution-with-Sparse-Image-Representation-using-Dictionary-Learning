function [ Y ] = MultiUserData( users, C, T, N )
%MultiUserData Summary of this function goes here
%   Detailed explanation goes here

Y = zeros(C,T,N);
mGroups = zeros(N,1);
uGroups = zeros(N,1);
mx = 1;
ux = 1;
for m = 1:N
    if mx > ceil(N/length(users))
        mx = 1;
        ux = ux + 1;
    end
    mGroups(m) = mx;
    uGroups(m) = users(ux);
    mx = mx + 1;
end

lastUser = 0;
lastM = 0;
t = 0;
for p = 1:N
    if uGroups(p) ~= lastUser
        lastUser = uGroups(p);
        f = sprintf('s%02d.mat', lastUser);
        disp(['Loaded data from ' f])
        load(f);
    end
    m = mGroups(p);
    fprintf('Movie Loaded %d\n', m);
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

