function l = lap(a)
% function l = lap(a)
%
% form laplacian matrix from an adjacency matrix

l = diag(sum(a)) - a;