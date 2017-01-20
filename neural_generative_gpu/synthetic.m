%% Triangle
syn_img = zeros(128,128);
for x = 1:128
    if x <= 64
        y = 128 - 1.6*x;
    else
        y = 1.6*x - (77);
    end
    syn_img(y:110, x) = 1;
end
figure(5)
clf
imshow(syn_img)
imwrite(syn_img, 'triangle.png')