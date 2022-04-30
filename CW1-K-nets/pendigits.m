% load data set
load pen_digits

% I. pendigits test data set

% build similarity matrix 
% based on euclidean distance
D=pdist(test_data);Dists=squareform(D);

% partition into 10 clusters
% utilizing a single-layer k-net.
tic;idx = knet(Dists, 190);toc

% utilizing a two-layer k-net
% tic;idx=knet(Dists, [6, 20]);toc;

% utilizing a two-layer k-net, with geodesic
% distances on the second layer
tic;idx=knet(Dists, [6, 20], 'geo');toc;

% validate partition against nmi metric
% nmi(idx,test_class)

% II. pendigits train data set 

% partition into 10 clusters
% utilizing a two-layer k-net.
tic;idx=knet(train_data, [3, 80]);toc;

% Same k-net architecture, different resolution
%tic;idx=knet(train_data, [5, 50]);toc;
%tic;idx=knet(train_data, [10, 25]);toc;

% validate partition against nmi metric
% nmi(idx,train_class)

% III. pendigits total data set 

% Two layer k-net
tic;idx=knet(total_data, [3 130]);toc;length(unique(idx))

% Same k-net architecture, different resolution
%tic;idx=knet(total_data, [10 35]);toc;length(unique(idx))

% With a three-layer k-net
%tic;idx=knet(total_data, [3 3 29]);toc;

% validate partition against nmi metric
% nmi(idx,total_class)

% Four layer k-net utilizing geodesic 
% distances in the last layer. 
tic;kns=knet(total_data, [2 2], 'struct');toc;length(unique(idx)) 
tic;idx=knet(kns, [3 20], 'geo');toc;length(unique(idx))

% validate partition against nmi metric
% nmi(idx,total_class)
