% load data set
load faces;  

% build similarity matrix based
% on vector of euclidean distances
Dists=squareform(D);

% Single layer k-net clustering
tic;idx=knet(Dists, 14);toc 

% The numnber of clusters is determined 
% by the number of medoids extracted.
disp(length(unique(idx)));

% validate partition against nmi metric
% nmi(idx,class)
