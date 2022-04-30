% load data set
load mnist_subset

% build similarity matrix 
% based on euclidean distance
D=pdist(data);Dists=squareform(D);

% Partition into 5 clusters initiating
% the exact mode from 9 clusters
tic;idx=knet(Dists, 160, 'exact', 5);toc;

% We have the same results, by initiating 
% the exact mode from any k value in the range
% 160 to 210.

% validate partition against nmi metric
% nmi(idx,class)

% Partition into 5 clusters with a two-layer
% k-network, utilizing approximate geodesic 
% distances in the second layer
tic;idx=knet(Dists, [7 25], 'geo');toc;

% validate partition against nmi metric
% nmi(idx,class) 

% Under different resolution:
%tic;idx=knet(Dists, [6 21], 'exact', 5,'geo');toc;length(unique(idx))
%tic;idx=knet(Dists, [5 21], 'exact', 5,'geo');toc;length(unique(idx))


