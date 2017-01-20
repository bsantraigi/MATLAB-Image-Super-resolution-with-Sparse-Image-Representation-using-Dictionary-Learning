function Plot_DistanceMat( points, f)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
figure(f)
clf
m = size(points,2);
dm = zeros(m,m);
for i = 1:m
    for j = 1:m
        dm(i,j) = norm(points(:,i) - points(:,j));
    end
end
imagesc(dm)

end

