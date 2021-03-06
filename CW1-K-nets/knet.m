function [idx, vals, Pnts_Scores] = knet(DN, k, varargin)
% KNET K-Network clustering.
% IDX = KNET(DATA, K), automatically partitions a data set,
% into a number of clusters based on the value of parameter K,
% that indicates the degree of the requested clustering resolution.
% IDX, is a vector, containing the cluster indices of each point.
% Each cluster is represented by an actual data point called medoid.
% Hence the cluster indices are the indices of the corresponding
% medoids in the data set. Depending on the available computational 
% resources, DATA can be, one of the followings:
% 1. a N-by-P pattern matrix,
% 2. a N-by-N similarity matrix, 
% 3. a k-network structure.
%
% K is a 1xL vector of integers, all with values larger than 1. The length
% of K indicates the number of KNET layers. The value(s) in K indicate 
% the clustering resolution of the corresponding layer. In current KNET 
% implementation, in cases that the data input form is a similarity matix
% the maximum number of layers is 2. In cases that a pattern matrix is 
% provided as input, the minimum number of layers is 2. 
%
% IDX = KNET(..., 'PARAM1', val1, 'PARAM2', val2, ...) specifies optional
% parameter name/value pairs to control KNET. Parameters are:
%
% 'exact' - Utilized in cases that we request from KNET a specific
%   number of clusters Cr. It should be used with a k value under 
%   which KNET: extracts the next smallest number of clusters than
%   the requested one Cr. 
%
% 'geo' - Engaged in cases we want to test a data set against
%   the existence of non-convex or non-linearly seperable clusters. 
%   The value of this parameter, indicates the number of nearest neighbors
%   that will be utilized in the calculation of geodesic distances. Default is 3.
%
% 'struct' - This switch should be used in cases that pattern matrix is the 
%   data input form. If engaged, KNET outputs a structure consisting of the 
%   input pattern matrix and IDX, instead of solely IDX. This structure can 
%   subsequently be used, instead of a pattern or similarity matrix in KNET. 
%   It can be utilized as an alternative for building multi-layer k-networks. 
% 
% 'metric' - Distance measure used to determine similarities among the N
%   M-dimensional points, in cases that pattern matrix is the input data
%   form. In the current KNET implementation can be one of the followings:
%    'euclidean'     - Euclidean distance (the default).
%    'sqeuclidean'   - Squared euclidean distance. 
%    'correlation'   - One minus the sample correlation between points
%                      (treated as sequences of values)
%    'hamming'       - Percentage of bits that differ (only suitable
%                      for binary data)
%
% Examples:
%
% Single layer k-net:
%
% d=rand(3000, 2);
% D=pdist(d);Dists=squareform(D);
% tic;idx=knet(Dists, 15);toc
% meds = unique(idx);hold  on;
% for i=1:length(meds)
%  plot(d(idx==meds(i),1),d(idx==meds(i),2),'-*','Color',rand(1,3));
% end
%
% Two-layers knet:
%
% data=rand(15000, 4);
% tic;idx=knet(data, [20 6]);toc;
% meds=unique(idx);
% for i=1:length(meds)
%     subplot(7, 7, i);plot(data(idx==meds(i),:)');
% end

% COPYRIGHT NOTICE
% K-net code (c) 2010-2013 Ioannis A. Maraziotis
%
% Further comments: 
%
% I. K-net, can also be applied on a 2xNxM cell structure. 
% The first member of the structure contains the indices 
% of the M nearest neighbors of every point, while the second the 
% corrsponding distances. 
%
% II. In cases that a pattern matrix is provided as input, the minimum 
% number of layers is 2, while if the length of K is 1, KNET automatically
% sets a value for the resolution of the first layer and the provided K is 
% utilized as the resolution of the second layer. 
%
% III. In cases that we want to test against the existence of highly irregular
% in terms of their size clusters, we can utilize the exact operational mode
% with a k value under which the standard operational mode of k-net extracts a 
% slightly larger number of clusters than the one requested. 
%
% IV. 2 additional k-net parameters can be set: 
% 'MaxIter' -  Maximum number of iterations allowed, in the final
%   phase of assignment. In multi-layers architectures, default is 1,  
%   while in single layer architecture is 50. 
%
% 'dstep' - Controlling the size of data components, the original data set is 
%   broken down to. It is used in cases that the form of data input is a pattern  
%   matrix. Default value is 300.

% Check the range of k values
if ~isempty(find(k<0)==1)
    fprintf('Error the value of the resolution parameter cannot be smaller than 1.\n');
    idx=[];vals=[];Pnts_Scores=[];
    return;
end

% Process input
inps = Process_Input(DN, k, varargin);
Dists=inps{1};Neighbs=inps{2};data=inps{3};
c=inps{5};kstep=inps{6};dstep=inps{7};
fprintf('dstep %d\n',dstep);
pidx=inps{8};metric=inps{9};
maxiters=inps{10};resolve=inps{11};
nlin=inps{12};knetstruct=inps{4};
initialize=1;
[Dists, Neighbs, K] = check_nans(Dists, Neighbs, k);

single_knet = 0;knet_serial_knet = 0;
parallel_knets_serial_knet = 0;

% Build a fixed k-net architecture, based on the data input form

if size(k, 2) == 1 && ( (size(Dists,1)==size(Dists,2)) || ~isempty(Neighbs))
    single_knet = 1;
    fprintf('Single Knet K=%d\n',size(k,2))
elseif size(k,2)>1 && (size(Dists,1)==size(Dists,2))
    knet_serial_knet = 1;
    fprintf('Serial Knet layer=%d\n',size(k,2))
elseif size(k,2)>1 && (size(Dists,1)~=size(Dists,2))
    parallel_knets_serial_knet = 1;
    maxiters=1;
    fprintf('Parallel Knet layer=%d\n',size(k,2))
elseif size(k,2)==1 && (size(Dists,1)~=size(Dists,2))
    parallel_knets_serial_knet = 1;
    k=[setk(size(Dists,1)), k];
    maxiters=1;
    fprintf('Parallel Knet2 %d %g\n',size(k,2),k)
end

%Perform K-net clustering

if single_knet
    fprintf('single_knet\n');
    [idx, vals, Pnts_Scores]= knet_cluster(Dists, Neighbs, K, kstep, resolve, c, maxiters);
elseif knet_serial_knet
    fprintf('knet_serial_knet Dists size=%d\n',size(Dists,1));
    fprintf('k1=%d\n',K(1));
    [tidx, vals]= knet_cluster(Dists, Neighbs, K(1), 1, 1, -1, maxiters);
    meds=unique(tidx);nDists=Dists(meds,meds);

    fprintf('\nknet_serial_knet size nDist=%d\n',size(nDists,1));
    fprintf('1 SerialKnet meds size=%d\n',size(meds,2));
    fprintf('1 Neighbs size=%d\n',size(Neighbs,2));
    fprintf('nlin=%d\n',nlin);
    if nlin > 0
        nDists = nlinmap(nDists, nlin);
    end
    fprintf('maxiters=%d\n',maxiters);
    fprintf('k2=%d\n',K(2));
    fprintf('c=%d\n',c);
    fprintf('resolve=%d\n',resolve);
    fprintf('kstep=%d\n',kstep);
    
    [idx, vals]= knet_cluster(nDists, Neighbs, K(2), kstep, resolve, c, maxiters);
    
    fprintf('knet_cluster idx\n');
%     disp(unique(idx));
    if nlin == 0
        fprintf('2 SerialKnet idx\n');
%         disp(idx);
        fmeds=unique(idx);
        fprintf('2 SerialKnet fmeds\n');
        disp(fmeds);
        for i=1:length(fmeds),idx(idx==fmeds(i))=meds(fmeds(i));end
        meds = unique(idx);
        fprintf('3 SerialKnet meds\n');
        disp(meds);
        [idx, vals] = knet_iterations(meds, Dists, Neighbs, maxiters);
%         disp(unique(idx)-1);
    else
        idx = assign_prior_labels(tidx, idx);
    end
elseif parallel_knets_serial_knet % data mode
    fprintf('parallel_knets_serial_knet\n');
    if initialize,
        fprintf('initialize\n');
        [Dists,inds] = initialize_data(Dists, metric);
    end
    [idx, vals] = partial_knet(Dists, k, dstep,  c, resolve, maxiters, kstep, metric);
    idx = reestate_data(initialize, idx, inds);
end
if ~isempty(pidx) && ~isempty(data)  && nlin==0
    fprintf('~isempty(pidx) && ~isempty(data)\n');
    pmeds=unique(pidx);meds=unique(idx);
    [mv,idx]=min(distfun(data, data(pmeds(meds),:), metric, 1),[],2);
    vals=sum(mv)/length(unique(idx));
    for i=1:length(meds),idx(idx==i)=pmeds(meds(i));end
elseif ~isempty(pidx)
    fprintf('~isempty(pidx)\n');
    if length(unique(idx))>1
        idx = assign_prior_labels(pidx, idx);
    else
        fprintf('Single cluster result. Decrease resolution value.\n');
    end
end
if knetstruct
    fprintf('knetstruct\n');
    if isstruct(DN)
        kns.data=DN.data;
    else
        kns.data=DN;
    end
    kns.prior=idx;
    kns.metric=metric;
    idx=kns;
end
end

function [idx, vals, Pnts_Scores] = knet_cluster(Dists, Neighbs, k, kstep,  resolve, c, maxiters)
idx=[];vals=[];Pnts_Scores=[];
fprintf('\n-----knet_cluster-----\n');
if ~iscell(Dists) && k>=size(Dists, 1), return; end
if c==-1
    fprintf('knet_cluster c=-1 Neighbs size=%d\n',size(Neighbs,1));
    [Scores, M] = PClusters(Dists, Neighbs, k,  resolve);
    [medoids, Pnts_Scores] = knet_select(Scores, M);
    fprintf('knet_cluster medoids size=%d\n',length(medoids));
%     disp(medoids);
    if length(unique(medoids)) == 1
        %fprintf('Single cluster result\n');
        idx=[];vals=[];
    else
        [idx, vals] = knet_iterations(medoids, Dists, Neighbs, maxiters);
    end
else
    fprintf('knet_cluster c=%d\n',c);
    [idx,vals] = sqknet(Dists, Neighbs, k, c, kstep, maxiters);
end
end

function [labels ,vals] = sqknet(Dists, Neighbs, k, c, kstep, maxiters)
fprintf('\n-----sqknet-----\n');
Scores=inf*ones(1,size(Dists,1));M=cell(1,size(Dists,1));
k0 = k;Meds = [];idx=zeros(1, size(Dists, 1));L=[];
if isempty(Neighbs)
    [SV, SI] = sort(Dists, 2);
    for k=k0:-kstep:1
        % Construction Phase
        for i=1:size(Dists, 1)
            sv = SV(i, :);si = SI(i, :);
            equal_distanced_members = find(sv<=sv(k));
            M{i}=si(1:equal_distanced_members(end));
            Scores(i) = (1/length(M{i}))*sum(sv(1:k));
        end;
        [~,si]=sort(Scores);
        % Selection Phase
        for i=1:size(Dists, 1)
            if sum(idx(M{si(i)}))==0
                Meds=[Meds si(i)];
                idx(M{si(i)})=1;
            end
        end
        L = [L length(Meds)];
        if length(Meds) >= c
            break;
        end
    end
    if c>L(end)
        c = L(end);
    end
else
    Scores=inf*ones(1,size(Dists,2));M=cell(1,size(Dists,2));
    k0 = k;Meds = [];idx=zeros(1, size(Dists, 2));L=[];
    for k=k0:-kstep:2
        % Construction Phase
        for i=1:size(Dists, 2)
            sv = Dists{i};si = Neighbs{i};
            equal_distanced_members = find(sv<=sv(k));
            M{i}=si(1:equal_distanced_members(end));
            Scores(i) = (1/length(M{i}))*sum(sv(1:k));
        end;
        [~,si]=sort(Scores);
        % Selection Phase
        for i=1:size(Dists, 2)
            if sum(idx(M{si(i)}))==0
                Meds=[Meds si(i)];
                idx(M{si(i)})=1;
            end
        end
        L = [L length(Meds)];
        if length(Meds) >= c
            break;
        end
    end
end
[labels,vals]=knet_iterations(Meds(1:c), Dists, Neighbs, maxiters);
end

function [Scores M] = PClusters(Dists, Neighbs, k,  resolve)
Scores = zeros(1,size(Dists,1));M = cell(1,size(Dists,1));
TScores = zeros(1,size(Dists,1));
fprintf('\n-----PClusters-----\n');
fprintf('PClusters size Dists=%d\n',size(Dists,1));
% Construction Phase: Construct the Pre-clusters from the Pre-medoids
if isempty(Neighbs)
    fprintf('PClusters Neighbs isempty\n');
    for i=1:size(Dists,1)
        [sv si] = sort(Dists(i,:));
        equal_distanced_members = find(sv<=sv(k));
        M{i}=si(1:equal_distanced_members(end));
        Scores(i) = (1/length(M{i}))*sum(sv(1:k));
        TScores(i) = (1/size(Dists, 1))*sum(sv(1:end));
    end
else % reduced information operation
    fprintf('PClusters Neighbs size=%d\n',size(Neighbs,1));
    for i=1:length(Dists) % If the provided nearest neighbos are not ordered uncomment the next line
        equal_distanced_members = find(Dists{i}<=Dists{i}(k(i)));
        M{i}=Neighbs{i}(1:equal_distanced_members(end));
        Scores(i) = (1/length(M{i}))*sum(Dists{i}(1:k(i)));
        TScores(i) = (1/size(Dists, 1))*sum(Dists{i}(1:end));
    end
end
if resolve
    fprintf('PClusters resolve=%d\n',resolve);
    Equal_Score_PCs = detect_instabilities(Scores);
    if ~isempty(Equal_Score_PCs)
        disp('PClusters Instability detected');
        Scores(Equal_Score_PCs)=TScores(Equal_Score_PCs);
    end
    fprintf('PClusters score=%d\n',size(Scores,2));
end
end

function Equal_Score_PCs = detect_instabilities(Scores)
    fprintf('\n-----detect_instabilities-----\n');
    Equal_Score_PCs=[];
    unq_Scores = unique(Scores);
    am=zeros(1,length(unq_Scores));%[];
    for i=1:length(unq_Scores)
        am(i) = length(find(Scores==unq_Scores(i)));
    end
    mult_scores = find(am>1);
    for i=1:length(mult_scores)
        cm=find(Scores==unq_Scores(mult_scores(i)));
        Equal_Score_PCs = [Equal_Score_PCs cm];
    end
end

function [medoids,si] = knet_select(Scores, M)
% Sort the Pre-Clusters based on their scores
    [~,si]=sort(Scores);
%     disp(si);
    tlabels = zeros(1, length(si));
    medoids = zeros(1, length(si));
    fprintf('\n-----knet_select-----\n');
    for i=1:length(si)
        if sum(tlabels(M{si(i)}))==0
            tlabels(M{si(i)})=1;
            medoids(si(i))=1;
        end
    end
    medoids = find(medoids==1);
%     disp(medoids);
    fprintf('%d\n',length(unique(medoids)));
end

function [vals, labels] = assign_labels(Dists, Neighbs, medoids)
M = size(Dists, 2);iM=M;ds=-1*ones(1, M);ls=ds;
% Initially find all patterns that one of the medoids is amongst their
% given nearest neighbors
fprintf('\n-----assign_labels-----\n');
pats2scan=1:M;nmedoids=medoids;c=1;
for j=1:100
    for i=1:M
        ip=ismember(Neighbs{pats2scan(i)}, nmedoids);
        fip = find(ip>0,1);
        if ~isempty(fip)
            ls(pats2scan(i))=Neighbs{pats2scan(i)}(fip);
            ds(pats2scan(i)) = Dists{pats2scan(i)}(fip);
        end
    end;
    traced=find(ds>-1);
    not_traced=find(ds==-1);
    if ~isempty(not_traced)
        M = length(not_traced);
        pats2scan=not_traced;
        c=c+1;
        if j>1
            for t=1:length(medoids)
                ls(ls==nmedoids(t))=medoids(t);
            end
        end
        nmedoids = [];
        % Check for c values
        nn_limit_reached = zeros(1, length(medoids));
        for t=1:length(medoids)
            if c> length(Neighbs{medoids(t)})
                nn_limit_reached(i) = 1;
            end
        end
        for t=1:length(medoids)
            if ~nn_limit_reached(t)
                nmedoids = [nmedoids Neighbs{medoids(t)}(c)];
            end
        end
        %if ~any(medoids, nmedoids)
        if length(intersect(medoids,nmedoids))==length(medoids)
            break;
        end
    else
        break;
    end
end
for t=1:length(medoids)
    ls(ls==nmedoids(t))=medoids(t);
end
if length(traced)~=iM
    not_traced = find(ds==-1);
    fprintf('Error:%3.5f percent of the patterns could not be assigned.\n', ((length(not_traced)/length(Dists)))*100);
    fprintf('Increase the amount of input data or reduce resolution parameter k.\n');
    ls = [];ds=[];
end
labels = ls;vals = ds;
end

function [labels, vals] = knet_iterations(meds, Dists, Neighbs, maxiters)
vals = -1*ones(1, maxiters);c=length(meds);
fprintf('\n-----knet_iterations-----\n');
if isempty(Neighbs)
    fprintf('isempty(Neighbs)\n');

    [val,labels] = min(Dists(meds,:));
    
    labels=meds(labels);
%     disp(labels);
    vals(1)=sum(val)/length(meds);
    fprintf('meds size=%d\n',length(meds));
%     disp(meds);
    fprintf('labels size=%d\n',length(labels));
    fprintf('vals %d\n',vals(1));
    
    for iters=1:maxiters
        for i = 1:c
            idx = (labels==meds(i));
            [~,tmp] = min(mean(Dists(idx,idx)));
            idx = find(idx);meds(i) = idx(tmp);
        end
        last = labels;
        [val,labels] = min(Dists(meds,:));
        labels=meds(labels);
        vals(iters+1)=sum(val)/length(meds);
        if ~any(labels ~= last)
            vals=vals(1:iters);
            break;
        end
    end
    vals=vals(vals>-1);
%     fprintf('labels\n');
%     disp(unique(labels));
else
    fprintf('not isempty(Neighbs)\n');
    [Vals, labels] = assign_labels(Dists, Neighbs, meds);
    if isempty(labels)
        return
    end
    vals = -1*ones(1, 100);
    vals(1) = sum(Vals)/length(meds);
    for iters=1:maxiters
        for i = 1:c
            idx = find(labels==meds(i));
            Vals = -1*ones(1,length(idx));
            
            for j=1:length(idx)
                ip=ismember(Neighbs{idx(j)}, idx);
                Vals(j) = mean(Dists{idx(j)}(ip==1)); % mean of distance of current cluster member from all other members
                [~,mi]=min(Vals);meds(i) = idx(mi);
            end;
        end
        last = labels;[Vals, labels] = assign_labels(Dists, Neighbs, meds);
        if isempty(labels)
            vals=vals(1:iters);
            return;    Scores=inf*ones(1,size(Dists,2));M=cell(1,size(Dists,2));
    k0 = k;Meds = [];idx=zeros(1, size(Dists, 2));L=[];
    for k=k0:-kstep:2
        % Construction Phase
        for i=1:size(Dists, 2)
            sv = Dists{i};si = Neighbs{i};
            equal_distanced_members = find(sv<=sv(k));
            M{i}=si(1:equal_distanced_members(end));
            Scores(i) = (1/length(M{i}))*sum(sv(1:k));
        end;
        [~,si]=sort(Scores);
        % Selection Phase
        for i=1:size(Dists, 2)
            if sum(idx(M{si(i)}))==0
                Meds=[Meds si(i)];
                idx(M{si(i)})=1;
            end
        end
        L = [L length(Meds)];
        if length(Meds) >= c
            break;
        end
    end
        end
        vals(iters+1)=sum(Vals)/length(meds);
        if ~any(labels ~= last)
            break;
        end
        if vals(iters+1) > vals(iters)
            fprintf('Msg: Not enough data to conclude iterative mode.\n');
            fprintf('Increase the amount of input data or reduce resolution parameter k.\n');
            labels=last; %labels=init_labels; %labels=last;
            vals(iters+1)=vals(iters);%vals=[];
            break;
        end
    end
    vals=vals(vals>-1);
end
end

function nlabels = assign_prior_labels(prior_labels, tlabels)
    fprintf('\n-----assign_prior_labels-----\n');
    fprintf('len prior %d\n',length(prior_labels));
    fprintf('len tlabels %d\n',length(tlabels));

    ul = unique(tlabels);
    prior_medoids=unique(prior_labels);
    
    fprintf('ul\n');
    disp(ul);
    fprintf('prior size=%d\n',length(prior_medoids));
    
    nlabels = zeros(1, length(prior_labels));
    for i=1:length(ul)
        cinds = find(tlabels==ul(i));
        fprintf('len cind %d\n',length(cinds));
        for j=1:length(cinds)
            nlabels(  prior_labels == prior_medoids(cinds(j))) = prior_medoids(ul(i));
        end;
    end;
end % Function

function [idx,val] = partial_knet(data, k, dstep,  c, resolve, maxiters, kstep, metric)
    fprintf('\n-----partial_knet-----\n');
    tdata=data;last=1:size(data,1);val=[];
    for lind=1:size(k, 2)-1
        fprintf('-----lind %d-----\n',lind);
        [Tidx,TInds] = knet_label(tdata, k(lind), kstep, dstep,resolve, maxiters, metric);
        if lind>1
            Tidx = assign_prior_labels(last, Tidx);
            disp(Tidx);
        end
        last = Tidx;
        pmeds = unique(Tidx);tdata=data(pmeds, :);
    end;
    STD_KNET_MAX_MEM_LIMIT = 6000;
    if length(pmeds) < STD_KNET_MAX_MEM_LIMIT
        fprintf('< STD_KNET_MAX_MEM_LIMIT pmeds %d-----\n',length(pmeds));
        idx = partial_knet_iters(data, pmeds, Tidx, k, kstep, resolve, c, maxiters, metric);
    else
        fprintf('Error: Larger resolution than reserved increase the number of layers in knet\n');
        idx=[];
    end
end % Function

function idx = partial_knet_iters(data, pmeds, Tidx,k, kstep, resolve, c, maxiters, metric)
    fprintf('\n-----partial_knet_iters-----\n');
    ndata=data(pmeds, :);
    Dists = distfun(ndata, ndata, metric, 0);
    idx=knet_cluster(Dists, [], k(end), kstep,  resolve, c, maxiters);
    fprintf('Number of medoids in the first layer: %d\n', length(idx));
    if length(idx)>=2
        idx = assign_prior_labels(Tidx, idx);
        pmeds=unique(idx);meds=pmeds;
        [mv,idx]=min(distfun(data, data(meds,:), metric, 1),[],2);
        zidx=zeros(1,length(idx));for i=1:length(meds),zidx(idx==i)=meds(i);end;idx=zidx;
        idx = part_knet_iters(data, idx, meds,maxiters,metric);
    else
        fprintf('Number of clusters less than 2. Reduce the resolution of at least the last layer.\n');
    end
end

function idx = part_knet_iters(data, idx, meds,maxiters,metric)
fprintf('\n-----part_knet_iters-----\n');
pidx=idx;mol=1500;
for q=1:maxiters
    disp(q);
    for c=1:length(meds)
        tidx=find(idx==meds(c));
        MV=[];MI=[];
        if length(tidx)>mol
            q=1:mol:length(tidx);p=[];for i=1:length(q)-1,p=[p; q(i), q(i+1)-1];end;p(end, 2)=length(tidx);
        else
            p=[1, length(tidx)];
        end
        for t=1:size(p,1)  % <- Prevent Memory Overhead if Oversized clusters
            [mv,mi]=min(mean(distfun(data(tidx,:), data(tidx(p(t,1):p(t,2)),:), 'seuclidean')));
            %[mv,mi]=min(mean(distfun(data(tidx,:), data(tidx(p(t,1):p(t,2)),:), 'euc')));
            temp=p(t,1):p(t,2);MV=[MV mv];MI=[MI temp(mi)];
        end
        [mv,mi]=min(MV);meds(c)=tidx(MI(mi));
    end
    [mv,idx]=min(distfun(data, data(meds,:), metric, 1),[],2);
    zidx=zeros(1,length(idx));for i=1:length(meds),zidx(idx==i)=meds(i);end;idx=zidx;
    if ~any(pidx~=idx), disp('ok');break; end
    pidx=idx;
end
end

function [Tidx,inds] = knet_label(data, k, kstep, dstep, resolve, maxiters, metric)
    fprintf('\n-----knet_label-----\n');
    Tidx=[];
    q=1:dstep:size(data,1);
    pairs=[];
    inds=[];
    fprintf('k=%d kstep=%d dstep=%d resolve=%d',k,kstep,dstep,resolve);
    for i=1:length(q)-1
        pairs=[pairs; q(i), q(i+1)-1];
    end;
    pairs(end, 2)=size(data,1);
    disp(pairs);
    % parfor i=1:size(pairs, 1)
    for i=1:size(pairs, 1)
        cRange=pairs(i, 1):pairs(i, 2);
        Dists = distfun(data(cRange, :), data(cRange, :), metric, 0);
        idx=knet_cluster(Dists, [], k, kstep, resolve, -1, maxiters);
        Tidx=[Tidx cRange(idx)];
        inds=[inds i*ones(1,length(idx))];
    end;
end

function D = distfun(X, C, dist, iter)
%DISTFUN: This function is part of the kmeans function
% if size(X)==size(C)
% X=X';[d,N] = size(X);X2 = sum(X.^2,1);D= repmat(X2,N,1)+repmat(X2',1,N)-2*X'*X;
% return;
% end
[n,p] = size(X);
D = zeros(n,size(C,1));
nclusts = size(C,1);
switch dist
    case 'euc'
        for i = 1:nclusts
            D(:,i) = (X(:,1) - C(i,1)).^2;
            for j = 2:p
                D(:,i) = D(:,i) + (X(:,j) - C(i,j)).^2;
            end
        end
        D=D.^0.5;
    case 'seuclidean'
        for i = 1:nclusts
            D(:,i) = (X(:,1) - C(i,1)).^2;
            for j = 2:p
                D(:,i) = D(:,i) + (X(:,j) - C(i,j)).^2;
            end
        end
    case 'hamming'
        for i = 1:nclusts
            D(:,i) = abs(X(:,1) - C(i,1));
            for j = 2:p
                D(:,i) = D(:,i) + abs(X(:,j) - C(i,j));
            end
            D(:,i) = D(:,i) / p;
            % D(:,i) = sum(abs(X - C(repmat(i,n,1),:)), 2) / p;
        end
    case 'cityblock'
        for i = 1:nclusts
            D(:,i) = abs(X(:,1) - C(i,1));
            for j = 2:p
                D(:,i) = D(:,i) + abs(X(:,j) - C(i,j));
            end
            % D(:,i) = sum(abs(X - C(repmat(i,n,1),:)), 2);
        end
end
end % function

function D = nlinmap(D, K)
N = size(D,1);
INF =  1000*max(max(D))*N;
[tmp, ind] = sort(D);
for i=1:N
    D(i,ind((2+K):end,i)) = INF;
end
D = min(D,D');    %% Make sure distance matrix is symmetric
tic;
for k=1:N
    D = min(D,repmat(D(:,k),[1 N])+repmat(D(k,:),[N 1]));
end
fprintf('nlinmap time: %g\n',toc-tic);
end

function [Dists,si] = initialize_data(Dists,metric)
d=distfun(Dists,mean(Dists),metric);
[mv,maxi]=max(d);
d=distfun(Dists,Dists(maxi,:),metric);
[sv,si]=sort(d);Dists=Dists(si,:);
end

function idx = reestate_data(initialize, idx, inds)
fprintf('\n-----reestate_data-----\n');
meds=unique(idx);
if initialize
    zidx=zeros(1,length(idx));
    for i=1:length(meds),
        zidx(inds(idx==meds(i)))=inds(meds(i));
    end;
    idx=zidx;
end
end

function [nDists, nNeighbs, K] = check_nans(Dists, Neighbs, k)
% Check if there are nan values
if isempty(Neighbs) && ~isempty(find(isnan(Dists)>0, 1))
    fprintf('Nan values detected resolving....\n');
    K = zeros(1, size(Dists, 2));
    fprintf('K = %\n',K);
    nNeighbs = cell(1, size(Dists, 1));
    nDists = cell(1, size(Dists, 1));
    for i=1:size(Dists, 2)
        no_nan_inds=find(isinf(Dists(i,:))==0);
        fprintf('[%] NoNanIds = %\n',i,no_nan_inds);
        if length(no_nan_inds)<k
            fprintf('Length %<%\n',length(no_nan_inds),k);
            [sv,si] = sort(Dists(i,no_nan_inds));
            fprintf('sv = %\nsi= %',sv,si);
            nNeighbs{i} = no_nan_inds(si);
            nDists{i} = sv;
            K(i) = length(no_nan_inds);
            fprintf('nNeighbs = %\n',nNeighbs{i});
        else
            fprintf('Length %<%\n',length(no_nan_inds),k);
            K(i) = k;
            [sv,si] = sort(Dists(i,no_nan_inds));
            fprintf('sv = %\nsi= %',sv,si);
            nNeighbs{i} = no_nan_inds(si);
            nDists{i} = sv;
            fprintf('nNeighbs = %\n',nNeighbs{i});
        end
    end;
elseif ~isempty(Neighbs)
    fprintf('Neighbs not empty\n');
    if size(k,1)==size(k,2)==1
        K=k(1)*ones(1, length(Dists));
        fprintf('K = %\n',K);
    end
    nDists=Dists;nNeighbs=Neighbs;
    fprintf('nDists = %\nnNeighbs = %',Dists,Neighbs);
else
    fprintf('Neighbs is empty\n');
    nDists=Dists;nNeighbs=[];
    K=k;
end
end

function k=setk(N)
if N <= 10000
    k=2;
elseif N>10000 && N<50000
    k=3;
elseif N>50000 && N <100000
    k=5;
elseif N>=100000 && N < 200000
    k=20;
end
end
% ********************************
function inps = Process_Input(DN, k, varargin)
inps = [];pidx = [];maxiters=100;dstep=300;
data=[];c=-1;kstep = 1;nlin=0;resolve = 1;
metric='euc';knetstruct=0;
varargin=varargin{1};i=1;
if isstruct(DN),
    pidx=DN.prior;meds=unique(pidx);metric=DN.metric;
    data=DN.data;
    if ~isempty(data)
        DN = distfun(data(meds,:), data(meds,:), metric, 0);
    end
end
while i<=length(varargin)
    if strcmp(varargin{i},'iterations')
        maxiters = varargin{i+1};i=i+1;
    elseif strcmp(varargin{i},'exact')
        c = varargin{i+1};i=i+1;
    elseif strcmp(varargin{i},'resolve')
        resolve = varargin{i+1};i=i+1;
    elseif strcmp(varargin{i},'geo')
        nlin=3;
        if length(varargin) > i
            t=varargin{i+1};
            if isnumeric(t), nlin = t; end;
        end
        i=i+1;
    elseif strcmp(varargin{i},'dstep')
        dstep = varargin{i+1};i=i+1;
    elseif strcmp(varargin{i},'kstep')
        kstep= varargin{i+1};i=i+1;
    elseif strcmp(varargin{i},'metric')
        metric = varargin{i+1};i=i+1;
    elseif strcmp(varargin{i}, 'prior')
        pidx = varargin{i+1};i=i+1;
    elseif strcmp(varargin{i}, 'struct')
        knetstruct=1;i=i+1;
    else i=i+1;
    end
end
if iscell(DN)==0
    Dists = DN;Neighbs=[];
else
    Dists = DN{1};Neighbs = DN{2};
end
inps{1}=Dists;inps{2}=Neighbs;inps{3}=data;
inps{4}=knetstruct;inps{5}=c;inps{6} = kstep;
inps{7}=dstep;inps{8}=pidx;inps{9}=metric;
inps{10}=maxiters;inps{11}=resolve;inps{12}=nlin;
end