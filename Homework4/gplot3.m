function h = gplot3(a,X)
% function h = gplot3(a,X)
%
% very hacked 3d gplot,
%
% Daniel A. Spielman, Sep 18, 2007

[ai,aj] = find(a);
an = [ai, aj, ones(size(ai))];
an = an';
an = an(:);
Y = X(an,:);
Y(3:3:end,:) = NaN;
plot3(Y(:,1),Y(:,2),Y(:,3))