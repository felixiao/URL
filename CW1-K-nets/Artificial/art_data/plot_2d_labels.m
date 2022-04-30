function [] = plot_2d_labels(data,labels, medoids, cols)

% This function operates 2D datasets, it plots a certain clustering scheme
% using different colours for every cluster.

meds_given = 1;
if isempty(medoids)
    medoids=unique(labels);
    meds_given = 0;
end


if nargin<4
    
    cc=jet(length(medoids));
    
    for i=1:length(medoids)
        col=rand(1,3);
        %plot(data(labels==medoids(i),1),data(labels==medoids(i),2), 'o', 'Color', cc(i, :), 'MarkerSize', 5);
        plot(data(labels==medoids(i),1),data(labels==medoids(i),2), 'o', 'Color', col, 'MarkerSize', 3);
        hold on;
        if meds_given == 1
            %plot(data(medoids(i),1),data(medoids(i),2),'*','MarkerEdgeColor',cc(i,:),'MarkerFaceColor',cc(i,:),'MarkerSize',10);
            plot(data(medoids(i),1),data(medoids(i),2),'*','MarkerEdgeColor',col,'MarkerFaceColor',col,'MarkerSize', 50);
            %text(data(medoids(i),1),data(medoids(i),2),num2str(i));
        end
    end
    
else
    for i=1:length(medoids)
        %col=rand(1,3);
        plot(data(labels==medoids(i),1),data(labels==medoids(i),2), 'o', ...
            'Color', cols(i), 'MarkerSize', 3);
        hold on;
        if meds_given == 1
            plot(data(medoids(i),1),data(medoids(i),2),'*',...
                'MarkerEdgeColor',cols(i),'MarkerFaceColor',cols(i),'MarkerSize',10);
        end
    end
    
end
