% Plot positions and momentums
for i = 1:length(momentums)
    q = positions{i};
    p = momentums{i};
    plot(q,p,'linewidth',2);
    hold on
    yLimits = get(gca,'YLim');  
end
set(gca, 'fontsize',20)