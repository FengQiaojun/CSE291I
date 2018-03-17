% a chain graph of 30 nodes

% Laplacian Matrix
a = diag(ones(1,29),1);
a = a + a';
la = diag(sum(a)) - a;

% eigenvector
[v,d] = eig(full(la));
d = diag(d);

% plot the first 20
for i =1:5
    figure;
    for j = 1:4
        % index = 31-((i-1)*4+j);
        index = (i-1)*4+j;       
        subplot(2,2,j);
        plot(v(:,index));
        hold on; 
        plot(v(:,index),'o');
        grid on;
        title(index);
    end
end