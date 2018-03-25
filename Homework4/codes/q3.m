% % Laplacian Matrix
% img_size = size(img,1);
% node_num = img_size^2;
% a = zeros(node_num,node_num);
% for i = 1:node_num
%     neigs_i = q2_neighbor_lattice(i,img_size);
%     for j = 1:length(neigs_i)
%         a(i,neigs_i(j)) = 1;
%     end
% end
% la = diag(sum(a)) - a;
% 
% % eigenvector
% [v,d] = eig(full(la));
% d = diag(d);

% directly read the stored first 500 eigenvectors
load('v_500.mat');
v = v_500;

img = imread('lena.png');
img_size = size(img,1);

n = 30;
% the projection on first n eigenvectors
% plot the first n
img_vec = double(reshape(img,img_size*img_size,1));
coeff = zeros(n,1);
for i =1:n
    coeff(i) = q3_projection(img_vec,v(:,i));
end

% linear combination of eigenvectors
img_vec_recon = zeros(size(img_vec));
for i =1:n
    img_vec_recon = img_vec_recon + coeff(i).*v(:,i);
end

figure;
%subplot(1,3,1)
imshow(imresize(img,5,'nearest'));
title('origin img')
figure;
%subplot(1,3,2)
img_recon = uint8(reshape(img_vec_recon,img_size,img_size));
imshow(imresize(img_recon,5,'nearest'));
title(sprintf('reconstructed with %d eigenvectors',n));
figure;
%subplot(1,3,3)
img_error = uint8(abs(double(img_recon)-double(img)));
imshow(imresize(img_error,5,'nearest'));
title('error img');
