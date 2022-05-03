function D = check_nans(X, C, dist, iter)
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
        fprintf('%s',D)
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