function D = check_nans(D, K)
fprintf('before nlinmap:\n');
disp(D);
N = size(D,1);
INF =  1000*max(max(D))*N;
[tmp, ind] = sort(D);
fprintf('Sort Ind:\n');
disp(ind);
for i=1:N
    D(i,ind((2+K):end,i)) = INF;
end
fprintf('middle nlinmap:\n');
disp(D);
D = min(D,D');    %% Make sure distance matrix is symmetric
for k=1:N
    D = min(D,repmat(D(:,k),[1 N])+repmat(D(k,:),[N 1]));
end
fprintf('After nlinmap:\n');
disp(D);
end