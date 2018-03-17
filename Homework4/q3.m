img = imread('img.png');

% Laplacian Matrix
img_size = size(img,1);
node_num = img_size^2;
a = zeros(node_num,node_num);
for i = 1:node_num
    neigs_i = q2_neighbor_lattice(i,img_size);
    for j = 1:length(neigs_i)
        a(i,neigs_i(j)) = 1;
    end
end
la = diag(sum(a)) - a;

% eigenvector
[v,d] = eig(full(la));
d = diag(d);

% the projection on first 20 eigenvectors
% plot the first 20
img_vec = double(reshape(img,img_size*img_size,1));
coeff = zeros(20,1);
for i =1:20
    coeff(i) = q3_projection(img_vec,v(:,i));
end

% linear combination of eigenvectors
img_vec_recon = zeros(size(img_vec));
for i =1:20
    img_vec_recon = coeff(i).*v(:,i);
end

img_recon = uint8(reshape(img_vec_recon,img_size,img_size));
figure;
imshow(img_recon);
figure;
imshow(abs(double(img_recon)-double(img)));