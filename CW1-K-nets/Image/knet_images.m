clear
clc

% load and plot image

I=imread('red_car.jpg');
% I=imread('horses.jpg');
% I=imread('sstar.jpg');

C = 4; % Number of clusters.

% build rgd dataset
[N1, N2, N3]=size(I);
data=reshape(double(I)/255, [N1*N2, N3]);

N = 1000; % Number of pixels

[r c]=size(data);
inds = 1:ceil(r/N):size(data,1);
rdata = data(inds, :);

[kdata, m] = pca(rdata', 'numcomponents', 2); 
knet_data = data*m;

% k-net Clustering
D = squareform(pdist(kdata));
tic;idx=knet(D, 100, 'exact', C);toc; % try other k values
length(unique(idx))
tidx = idx;

meds=inds(unique(idx));
[mv, idx]=min(pdist2(knet_data(meds,:), knet_data));
meds=unique(idx);

% Plot results
p=2;
if C > 4
    p = 3;
end
figure('units','normalized','outerposition',[0 0 1 1])
for i=1:C
    tinds=find(idx==meds(i));
    zdata=0.2*ones(size(data));
    zdata(tinds,:)=data(tinds,:);
    zi=reshape(zdata, N1, N2, N3);
    subplot(p,p,i);imagesc(zi);axis image;axis off;title(i);
end

