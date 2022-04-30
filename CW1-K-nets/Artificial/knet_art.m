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
fprintf('Partitioning %d points, ',size(data,1));
D=pdist(data);Dists=squareform(D);

tic;idx=knet(Dists, 45);

% Same results with different k-net architectures:
% tic;idx=knet(data, [3,15]);toc;
% tic;idx=knet(data, [3,3,5]);toc;

fprintf('into %d clusters.\n',length(unique(idx)));
toc;subplot(3,3,1);fprintf('\n\n');
plot_2d_labels(data, idx, []);axis([2 30 3 31]);

load data7
fprintf('Partitioning %d points, ',size(data,1));
D=pdist(data);Dists=squareform(D);

tic;idx=knet(Dists, 250);

% Same results with different k-net architectures: 
%tic;idx=knet(data, [3, 60]);toc;
%tic;idx=knet(data, [3, 3, 15]);toc;

fprintf('into %d clusters.\n',length(unique(idx)));
toc;subplot(3,3,2);fprintf('\n\n');
plot_2d_labels(data, idx, [], 'mbgycrk');axis([0 1 0.3 1]);

load data4
fprintf('Partitioning %d points, ',size(data,1));
D=pdist(data);Dists=squareform(D);

tic;idx=knet(Dists, 150, 'exact', 4);

% Same results with different k-net architectures: 
%tic;idx=knet(data, [3, 50], 'exact', 4);toc;
% tic;idx=knet(data, [3,3,15], 'exact', 4);toc;

fprintf('into %d clusters.\n',length(unique(idx)));
toc;subplot(3,3, 3);fprintf('\n\n');
plot_2d_labels(data, idx, [], 'rgbk');

load data3
fprintf('Partitioning %d points, ',size(data,1));
D=pdist(data);Dists=squareform(D);
tic;idx=knet(Dists, [5, 90], 'geo');
fprintf('into %d clusters.\n',length(unique(idx)));
toc;subplot(3,3,4);fprintf('\n\n');
plot_2d_labels(data, idx, [], 'rkb');
axis([0 0.88 0.1 1.2]);

load ncircles
fprintf('Partitioning %d points, ',size(data,1));
D=pdist(data);Dists=squareform(D);
tic;idx=knet(Dists, [3, 120], 'geo');
fprintf('into %d clusters.\n',length(unique(idx)));
toc;subplot(3,3,5);fprintf('\n\n');
plot_2d_labels(data, idx, [],'rb');
axis([-20 40 -30 30]);

load data50
fprintf('Partitioning %d points, ',size(data,1));
tic;idx=knet(data, [5, 15]);
fprintf('into %d clusters.\n',length(unique(idx)));
toc;subplot(3, 3, 6);fprintf('\n\n');
plot_2d_labels(data, idx, [])
axis([0 69000 0 69000]);
 
load noisy_spiral
fprintf('Partitioning %d points, ',size(data,1));
tic;kns=knet(data, [3, 3], 'struct');
idx=knet(kns, [2, 200], 'geo', 2);  
fprintf('into %d clusters.\n',length(unique(idx)));
toc;subplot(3, 3, 7);fprintf('\n\n');
plot_2d_labels(data, idx, [],'rb');
axis([-50 50 -55 55]);

load birch_sin
fprintf('Partitioning %d points, ',size(data,1));

tic;idx=knet(data, [20, 30]);

% Same results with different k-net resolution:
% tic;idx=knet(data, [15, 25]);
% tic;idx=knet(data, [30, 17]);
%tic;idx=knet(data, [40, 12]);

fprintf('into %d clusters.\n',length(unique(idx)));
toc;subplot(3,3,8);fprintf('\n\n');
plot_2d_labels(data, idx, []);

load birch_grid
fprintf('Partitioning %d points, ',size(data,1));
tic;idx=knet(data, [20, 30]);
fprintf('into %d clusters.\n',length(unique(idx)));
toc;subplot(3,3,9);fprintf('\n\n');
plot_2d_labels(data, idx, []);