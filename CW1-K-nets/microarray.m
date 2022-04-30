% load data set
load microarray;

% Partition the data set into 15 clusters.
tic;idx=knet(data, [3, 30]);toc;

% Detect the number of clusters. 
disp(length(unique(idx)));

% Partition the data set into 10 clusters, 
% initiating the exact operational mode from
% a k value under which standard k-net mode 
% extracts 15 clusters.
tic;idx=knet(data, [3, 30], 'exact', 10);toc;

% Locate the medoids of the 10 clusters.
meds=unique(idx);

% Build Heatmap of the cluster that best 
% discriminates among the two experimental
% conditions. 
h=HeatMap([data(find(idx==meds(4)),:)], ...
    'ColumnLabels', samples)

