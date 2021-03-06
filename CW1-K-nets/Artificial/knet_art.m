% Application of K-net into aritificial, 2-dimensional
% data sets, illustrating the properties of the 
% clustering algorithm.

% Maraziotis Ioannis
% 2010 - 2012

figure;clf;clc

if ~exist('plot_2d_labels.m', 'file')
    disp('Function: plot_2d_labels is requested for the visualization of the results.');
%     break;
end

load data31
fprintf('data31 Partitioning %d points, ',size(data,1));
D=pdist(data);Dists=squareform(D);
% tic;idx=knet(Dists, 45);
% writematrix(idx.toarray(),'data31_true.csv')

% Same results with different k-net architectures:
tic;idx=knet(data, [3,15]);toc;
% tic;idx=knet(data, [3,3,5]);toc;
% ids=idx-1;
% writematrix(ids,'data31_[3,15]_true.csv')
fprintf('into %d clusters.\n',length(unique(idx)));
toc;subplot(3,3,1);fprintf('\n\n');
plot_2d_labels(data, idx, []);axis([2 30 3 31]);
% 
% load data7
% fprintf('data7 Partitioning %d points, ',size(data,1));
% D=pdist(data);Dists=squareform(D);
% 
% tic;idx=knet(Dists, 250);
% % writematrix(idx,'data7_true.csv')
% % Same results with different k-net architectures: 
% %tic;idx=knet(data, [3, 60]);toc;
% %tic;idx=knet(data, [3, 3, 15]);toc;
% 
% fprintf('into %d clusters.\n',length(unique(idx)));
% toc;subplot(3,3,2);fprintf('\n\n');
% plot_2d_labels(data, idx, [], 'mbgycrk');axis([0 1 0.3 1]);
% 
% load data4
% fprintf('data4 Partitioning %d points, ',size(data,1));
% D=pdist(data);Dists=squareform(D);
% 
% % tic;idx=knet(Dists, 150, 'exact', 4);
% 
% % Same results with different k-net architectures: 
% tic;idx=knet(data, [3, 50], 'exact', 8);toc;
% % tic;idx=knet(data, [3,3,15], 'exact', 4);toc;
% writematrix(idx,'data4_[3,50]_exact8_true.csv')
% fprintf('into %d clusters.\n',length(unique(idx)));
% toc;subplot(3,3, 3);fprintf('\n\n');
% plot_2d_labels(data, idx, []);

% load data3
% fprintf('data3 Partitioning %d points, ',size(data,1));
% D=pdist(data);Dists=squareform(D);
% % tic;idx=knet(Dists, [5, 90], 'geo');
% tic;idx=knet(Dists, [5, 90],'exact',5);
% writematrix(idx,'data3_exact5_true.csv')
% 
% fprintf('into %d clusters.\n',length(unique(idx)));
% toc;subplot(3,3,4);fprintf('\n\n');
% plot_2d_labels(data, idx, [], 'rkbgm');
% axis([0 0.88 0.1 1.2]);

% load ncircles
% fprintf('ncircles Partitioning %d points, ',size(data,1));
% D=pdist(data);Dists=squareform(D);
% % tic;idx=knet(Dists, [3, 120], 'geo');
% tic;idx=knet(Dists, [3, 120]);
% ids = idx-1
% % writematrix(ids,'ncircles_geo0_true.csv')
% % 
% fprintf('into %d clusters.\n',length(unique(idx)));
% toc;subplot(3,3,5);fprintf('\n\n');
% plot_2d_labels(data, idx, [],'rb');
% axis([-20 40 -30 30]);
% 
% load data50
% fprintf('Data50 Partitioning %d points, ',size(data,1));
% tic;idx=knet(data, [5, 15]);
% % writematrix(idx,'data50_true.csv')
% 
% fprintf('into %d clusters.\n',length(unique(idx)));
% toc;subplot(3, 3, 6);fprintf('\n\n');
% plot_2d_labels(data, idx, [])
% axis([0 69000 0 69000]);
%  
% load noisy_spiral
% fprintf('noisy_spiral Partitioning %d points, ',size(data,1));
% tic;kns=knet(data, [3, 3], 'struct');
% idx=knet(kns, [2, 200], 'geo', 2);
% fprintf('into %d clusters.\n',length(unique(idx)));
% toc;subplot(3, 3, 7);fprintf('\n\n');
% plot_2d_labels(data, idx, [],'rb');
% axis([-50 50 -55 55]);
% 
% load birch_sin
% fprintf('birch_sin Partitioning %d points, ',size(data,1));
% 
% tic;idx=knet(data, [20, 30]);
% % writematrix(idx,'birch_sin_true.csv')
% % Same results with different k-net resolution:
% % tic;idx=knet(data, [15, 25]);
% % tic;idx=knet(data, [30, 17]);
% %tic;idx=knet(data, [40, 12]);
% 
% fprintf('into %d clusters.\n',length(unique(idx)));
% toc;subplot(3,3,8);fprintf('\n\n');
% plot_2d_labels(data, idx, []);
% 
% load birch_grid
% fprintf('birch_grid Partitioning %d points, ',size(data,1));
% tic;idx=knet(data, [20, 30]);
% % writematrix(idx,'birch_grid_true.csv')
% 
% fprintf('into %d clusters.\n',length(unique(idx)));
% toc;subplot(3,3,9);fprintf('\n\n');
% plot_2d_labels(data, idx, []);