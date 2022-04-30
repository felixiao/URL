% load data set
load text;

% build similarity matrix based
% on vector of euclidean distances
Dists=squareform(D);

% Single layer k-net clustering
tic;idx=knet(Dists, 250);toc 

% The numnber of clusters is determined 
% by the number of extracted medoids
disp(length(unique(idx)));

% validate partition against nmi metric
% nmi(idx,class) 

% Two-layer k-net clustering, utilizing
% approximate geodesic distances in the
% second layer

tic;idx=knet(Dists, [5, 35], 'geo');toc 

% validate partition against nmi metric
% nmi(idx,class) 
