% a lattice of 30 by 30

% Laplacian Matrix
a = zeros(900,900);
for i = 1:900
    neigs_i = q2_neighbor_lattice(i,30);
    for j = 1:length(neigs_i)
        a(i,neigs_i(j)) = 1;
    end
end
la = diag(sum(a)) - a;

% eigenvector
[v,d] = eig(full(la));
d = diag(d);


% plot the first 20
for i =1:5
    figure;
    for j = 1:4
        % index = 901-((i-1)*4+j);
        index = (i-1)*4+j;       
        subplot(2,2,j);
        colormap(gray);
        imagesc(reshape(v(:,index),30,30)); 
        title(index);
    end
end