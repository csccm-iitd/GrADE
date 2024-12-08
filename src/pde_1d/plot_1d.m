%% layer__bt_test_1d
a = load('C:\Users\yash kumar\PycharmProjects\gcn\src\pde_1d\weights_results\burgers1d\layer__bt_test_1d\layer__bt_test_1d.csv');
clr = {'y','m','c','r','g','b'};
figure;

t = 4:4:20;
y1 = a(1,:); c1 = ['-o', clr{1}];
y2 = a(2,:); c2 = ['-s', clr{2}];
y3 = a(3,:); c3 = ['-*', clr{3}];
% y4 = a(4,:); c4 = ['-p', clr{4}];
% y5 = a(5,:); c5 = ['-+', clr{5}];
% y6 = a(6,:); c6 = ['-d', clr{6}];

plot(...
    t,y1,c1,... 
    t,y2,c2,... 
    t,y3,c3,... 
    'MarkerSize',12,...
    'LineWidth',2);
% p(1).MarkerEdgeColor = 'green';
% plot(t,y,'DurationTickFormat','mm:ss','-o')
% tl = ['no. of basis =', int2str(n_B(i))]
% title(tl)


set(gcf,'color','w')
% set(gca, 'YScale', 'log')
set(gca,'LineWidth',2/2,'FontSize',28/2,'FontWeight','normal','FontName','Times')
set(get(gca,'xlabel'),'String','rollout time in testing ','FontSize',32/2','FontWeight',...
'bold','FontName','Times','Interpreter','tex')
set(get(gca,'ylabel'),'String','Prediction error','FontSize',32/2','FontWeight',...
'bold','FontName','Times','Interpreter','tex')
set(gcf,'Position',[1 1 round(1000/2) round(1000/2)])
h = legend({'PointNetLayer', 'SpiderConv', 'GATConv'},'location','NW','FontSize',28/2,'FontWeight','normal','FontName','Times');
% export_fig(gcf,'scatter1.eps','-eps','-r300')
export_fig(gcf,'layer__bt_test_1d.pdf','-pdf','-r300')

% axis([.8 11 0.006 0.6])
% yticks([0.008 0.012 0.018 0.1 0.4])
xticks(t)
% xticklabels({'0','1','2','5','10'})

% axis([16 80 0,1.5])
% bb = linspace(0,1.5,15);
% yticks(bb);
% print -depsc PrMvsS
% export_fig(gcf,'PrOnlineNoise.pdf','-pdf','-r300')

%% lit_train__lit_test_1d

exp = "lit_train__lit_test_1d";
path = project_dir() + "pde_1d\weights_results\burgers1d\" + exp + "\" + exp + ".csv";
data = load(path);

legends = {'lit train-2', 'lit train-3', 'lit train-4', 'lit train-5'};
lgnd_loc = 'NW';
x_tick = 3:3:21; y_tick = 0; x_tick_lables = 0; yscale_log = 0;
x_label = 'Time index'; y_label = 'Prediction error';

name = project_dir() + "pde_1d\weights_results\burgers1d\" + exp + "\" + exp + ".pdf";

plot_graph(data, legends, lgnd_loc, x_tick, y_tick, name, x_label, y_label, axis_value, x_tick_lables, yscale_log, 1);

%% tr_size__lit_test_1d
exp = "tr_size__lit_test_1d";
path = project_dir() + "pde_1d\weights_results\burgers1d\" + exp + "\" + exp + ".csv";
data = load(path);

legends = {'train size-30', 'train size-60', 'train size-90', 'train size-120'};
lgnd_loc = 'NW';
x_tick = 3:3:21; y_tick = 0; x_tick_lables = 0; yscale_log = 0;
x_label = 'Time index'; y_label = 'Prediction error';

name = project_dir() + "pde_1d\weights_results\burgers1d\" + exp + "\" + exp + ".pdf";

plot_graph(data, legends, lgnd_loc, x_tick, y_tick, name, x_label, y_label, axis_value, x_tick_lables, yscale_log, 1);

%% B__bt_test__ad
a = load('C:\Users\yash kumar\PycharmProjects\gcn\src\pde_1d\weights_results\burgers1d\B__bt_test__ad.csv');
clr = {'y','m','c','r','g','b'};
figure;

t = [5, 10, 15, 20];
y = rand(1,5);
y1 = a(1,:); c1 = ['-o', clr{1}];
y2 = a(2,:); c2 = ['-s', clr{2}];
y3 = a(3,:); c3 = ['-*', clr{3}];
y4 = a(4,:); c4 = ['-p', clr{4}];
% y5 = a(5,:); c5 = ['-+', clr{5}];
% y6 = a(6,:); c6 = ['-d', clr{6}];

plot(...
    t,y1,c1,... 
    t,y2,c2,... 
    t,y3,c3,... 
    t,y4,c4,... 
    'MarkerSize',12,...
    'LineWidth',2);
% p(1).MarkerEdgeColor = 'green';
% plot(t,y,'DurationTickFormat','mm:ss','-o')
% tl = ['no. of basis =', int2str(n_B(i))]
% title(tl)


set(gcf,'color','w')
% set(gca, 'YScale', 'log')
set(gca,'LineWidth',2,'FontSize',28,'FontWeight','normal','FontName','Times')
set(get(gca,'xlabel'),'String','rollout time in testing ','FontSize',32','FontWeight',...
'bold','FontName','Times','Interpreter','tex')
set(get(gca,'ylabel'),'String','Prediction error','FontSize',32','FontWeight',...
'bold','FontName','Times','Interpreter','tex')
set(gcf,'Position',[1 1 round(1000) round(1000)])
h = legend({'Chebychev', 'Fourier', 'VanillaRBF', 'GaussianRBF', 'MultiquadRBF', 'Polynomial'},'location','NW','FontSize',28,'FontWeight','normal','FontName','Times');
% export_fig(gcf,'scatter1.eps','-eps','-r300')
export_fig(gcf,'B__bt_test__ad.pdf','-pdf','-r300')

% axis([.8 11 0.006 0.6])
% yticks([0.008 0.012 0.018 0.1 0.4])
xticks([5, 10, 15, 20])
% xticklabels({'0','1','2','5','10'})

% axis([16 80 0,1.5])
% bb = linspace(0,1.5,15);
% yticks(bb);
% print -depsc PrMvsS
% export_fig(gcf,'PrOnlineNoise.pdf','-pdf','-r300')

%% adptive__none
a = load('C:\Users\yash kumar\PycharmProjects\gcn\src\pde_1d\weights_results\burgers1d\adptive__none.csv');
clr = {'y','m','c','r','g','b'};
figure;

t = [5, 17, 20, 26, 35];%0:seconds(30):minutes(3);
y = rand(1,5);
y1 = a(1,:); c1 = ['-o', clr{5}];
y2 = a(2,:); c2 = ['-s', clr{2}];
% y3 = a(3,:); c3 = ['-*', clr{3}];
% y4 = a(4,:); c4 = ['-p', clr{4}];
% y5 = a(5,:); c5 = ['-+', clr{5}];
% y6 = a(6,:); c6 = ['-d', clr{6}];

plot(...
    t,y1,c1,... 
    t,y2,c2,... 
    'MarkerSize',12,...
    'LineWidth',2);
% p(1).MarkerEdgeColor = 'green';
% plot(t,y,'DurationTickFormat','mm:ss','-o')
% tl = ['no. of basis =', int2str(n_B(i))]
% title(tl)


set(gcf,'color','w')
% set(gca, 'YScale', 'log')
set(gca,'LineWidth',2,'FontSize',28,'FontWeight','normal','FontName','Times')
set(get(gca,'xlabel'),'String','rollout time in testing ','FontSize',32','FontWeight',...
'bold','FontName','Times','Interpreter','tex')
set(get(gca,'ylabel'),'String','Prediction error','FontSize',32','FontWeight',...
'bold','FontName','Times','Interpreter','tex')
set(gcf,'Position',[1 1 round(1000) round(1000)])
h = legend({'non-adaptive','adaptive'},'location','NW','FontSize',28,'FontWeight','normal','FontName','Times');
% export_fig(gcf,'scatter1.eps','-eps','-r300')
export_fig(gcf,'adptive__none.pdf','-pdf','-r300')

% axis([.8 11 0.006 0.6])
% yticks([0.008 0.012 0.018 0.1 0.4])
xticks(t)
% xticklabels({'0','1','2','5','10'})

% axis([16 80 0,1.5])
% bb = linspace(0,1.5,15);
% yticks(bb);
% print -depsc PrMvsS
% export_fig(gcf,'PrOnlineNoise.pdf','-pdf','-r300')

%% 
function [] = plotBarStackGroups(stackData, groupLabels)
% Plot a set of stacked bars, but group them according to labels provided.
%
% Params: 
%      stackData is a 3D matrix (i.e., stackData(i, j, k) => (Group, Stack, StackElement)) 
%      groupLabels is a CELL type (i.e., { 'a', 1 , 20, 'because' };)
%
% Copyright 2011 Evan Bollig (bollig at scs DOT fsu ANOTHERDOT edu
%
% 
NumGroupsPerAxis = size(stackData, 1);
NumStacksPerGroup = size(stackData, 2);
% Count off the number of bins
groupBins = 1:NumGroupsPerAxis;
MaxGroupWidth = 0.65; % Fraction of 1. If 1, then we have all bars in groups touching
groupOffset = MaxGroupWidth/NumStacksPerGroup;
figure
    hold on; 
for i=1:NumStacksPerGroup
    Y = squeeze(stackData(:,i,:));
    
    % Center the bars:
    
    internalPosCount = i - ((NumStacksPerGroup+1) / 2);
    
    % Offset the group draw positions:
    groupDrawPos = (internalPosCount)* groupOffset + groupBins;
    
    h(i,:) = bar(Y, 'stacked');
    set(h(i,:),'BarWidth',groupOffset);
    set(h(i,:),'XData',groupDrawPos);
end
hold off;
set(gca,'XTickMode','manual');
set(gca,'XTick',1:NumGroupsPerAxis);
set(gca,'XTickLabelMode','manual');
set(gca,'XTickLabel',groupLabels);

export_fig(gcf,'time__epoch.pdf','-pdf','-r300')
end 

%%

function project_dir = project_dir()
project_dir = "G:\My Drive\Colab Notebooks\gnode_pde\src\";
end


function plot_graph(data, legends, lgnd_loc, x_tick, y_tick, name, x_label, y_label, axis_value, x_tick_lables, yscale_log, m_legend)

disp(data);
clr = {'b','m','g','r','c','y'};
shapes = {'-o', '-s', '-*', '-p', '-+', '-d'};
figure;
% n_x_ticks = length(data);
n_line_graph = length(data(:, 1));
t = x_tick;

for i=1:n_line_graph
        
y_ = data(i,:); c_ = [shapes{i}, clr{i}];
p(i) = plot(t, y_, c_, 'MarkerSize',12, 'LineWidth',2);
hold on;
end
hold off

set(gcf,'color','w')
if yscale_log == 1
set(gca, 'YScale', 'log');
end
set(gca,'LineWidth',2/2,'FontSize',28/2,'FontWeight','normal','FontName','Times')
set(get(gca,'xlabel'),'String', x_label,'FontSize',32/2','FontWeight',...
'bold','FontName','Times','Interpreter','tex')
set(get(gca,'ylabel'),'String',y_label,'FontSize',32/2','FontWeight',...
'bold','FontName','Times','Interpreter','tex')
set(gcf,'Position',[1 1 round(1000/2) round(1000/2)])

% if exist('m_legend','var')
%     h1 = legend(p(1:3),legends(1:3),'FontSize',28/3,'FontWeight','normal','FontName','Times','Orientation','Horizontal');
%     set(h1, 'Position', [0.4,0.87,0.5,0.035]);
%     a=axes('position',get(gca,'position'),'visible','off');
%     h2 = legend(a, p(4:6),legends(4:6),'FontWeight','normal','FontName','Times','Orientation','Horizontal');
%     set(h2, 'Position', [0.4,0.87-0.038,0.5,0.035]);
% else 
h = legend(legends,'location',lgnd_loc,'FontSize',28/3,'FontWeight','normal','FontName','Times','Orientation','Vertical');
% end

% export_fig(gcf,'scatter1.eps','-eps','-r300')
export_fig(gcf,name,'-pdf','-r300')

if axis_value ~= 0
    axis(axis_value)
end
if y_tick ~= 0
    yticks(y_tick)
end
if ~isnumeric(x_tick_lables)
    xticklabels(x_tick_lables)
end
xticks(t)

end

