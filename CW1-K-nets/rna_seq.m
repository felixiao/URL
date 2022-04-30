% load data
load rna_seq;

% Partition the data set in 445 clusters
tic;idx=knet(data, [5 6]);toc;

% Detect the medoids
meds=unique(idx);length(meds)

% Plot a Heatmap of some of the clusters.
alist=[203, 111 31, 16, 275 265 195 84];
data=data(:,7:end);gdata=[];genes_inds=[];
for i=1:length(alist)
gdata=[gdata; data(find(idx==meds(alist(i))),:)];
genes_inds = [genes_inds find(idx==meds(alist(i)))];
end
h=HeatMap(gdata)
set(h, 'ColumnLabels', samples(7:end), ...
    'RowLabels', genes(genes_inds));


