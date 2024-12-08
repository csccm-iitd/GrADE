%%
% a1 = load('C:\Users\yash kumar\PycharmProjects\gcn\src\PDEs\Results\Delta{u}Delta{u}\PointNetLayer\Chebychev\n_B1.csv');
% a2 = load('C:\Users\yash kumar\PycharmProjects\gcn\src\PDEs\Results\Delta{u}Delta{u}\PointNetLayer\Chebychev\n_B2.csv');
% a3 = load('C:\Users\yash kumar\PycharmProjects\gcn\src\PDEs\Results\Delta{u}Delta{u}\PointNetLayer\Chebychev\n_B3.csv');
% a4 = load('C:\Users\yash kumar\PycharmProjects\gcn\src\PDEs\Results\Delta{u}Delta{u}\PointNetLayer\Chebychev\n_B5.csv');
% a5 = load('C:\Users\yash kumar\PycharmProjects\gcn\src\PDEs\Results\Delta{u}Delta{u}\PointNetLayer\Chebychev\n_B10.csv');
% at = {a1, a2, a3, a4, a5};
% 
% % figure;
% clr = {'y','m','c','r','g','b'};
% n_B = [1, 2, 3, 5, 10];
% for i=1:5%00
% figure;
% 
% a = at{1,i};
% t = [2, 6, 10, 18, 26, 38];%0:seconds(30):minutes(3);
% y = rand(1,5);
% y1 = a(1,:); c1 = ['-o', clr{i}];
% y2 = a(2,:); c2 = ['-s', clr{i}];
% y3 = a(3,:); c3 = ['-*', clr{i}];
% y4 = a(4,:); c4 = ['-p', clr{i}];
% y5 = a(5,:); c5 = ['-+', clr{i}];
% y6 = a(6,:); c6 = ['-d', clr{i}];
% 
% plot(...
%     t,y1,c1,... 
%     t,y2,c2,... 
%     t,y3,c3,... 
%     t,y4,c4,... 
%     t,y5,c5,...
%     t,y6,c6,... 
%     'MarkerSize',12,...
%     'LineWidth',2);
% % p(1).MarkerEdgeColor = 'green';
% % plot(t,y,'DurationTickFormat','mm:ss','-o')
% tl = ['no. of basis =', int2str(n_B(i))]
% title(tl)
% 
% 
% set(gcf,'color','w')
% % set(gca, 'YScale', 'log')
% set(gca,'LineWidth',2,'FontSize',28,'FontWeight','normal','FontName','Times')
% set(get(gca,'xlabel'),'String','batch_time','FontSize',32','FontWeight',...
% 'bold','FontName','Times','Interpreter','tex')
% set(get(gca,'ylabel'),'String','Prediction error','FontSize',32','FontWeight',...
% 'bold','FontName','Times','Interpreter','tex')
% set(gcf,'Position',[1 1 round(1000) round(1000)])
% h = legend({'5', '15', '25', '35', '45', '50'},'location','NE','FontSize',28,'FontWeight','normal','FontName','Times');
% % export_fig(gcf,'scatter1.eps','-eps','-r300')
% export_fig(gcf,'PrOnlineSeq.pdf','-pdf','-r300')
% 
% % axis([.8 11 0.006 0.6])
% % yticks([0.008 0.012 0.018 0.1 0.4])
% xticks([2, 6, 10, 18, 26, 38])
% % xticklabels({'0','1','2','5','10'})
% 
% % axis([16 80 0,1.5])
% % bb = linspace(0,1.5,15);
% % yticks(bb);
% % print -depsc PrMvsS
% % export_fig(gcf,'PrOnlineNoise.pdf','-pdf','-r300')
% % hold on
% end

%% layer__lit_test_2d
exp = "layer__lit_test_2d";
path = project_dir() + "pde_2d\weights_results\burgers2d\" + exp + "\" + exp + ".csv";
data = load(path);

legends = {'GATConv', 'SpiderConv'};
lgnd_loc = 'NW';
x_tick = 5:5:30; y_tick = 0; x_tick_lables = 0; yscale_log = 0;
axis_value = 0; x_label = 'Time index'; y_label = 'Prediction error';

name = project_dir() + "pde_2d\weights_results\burgers2d\" + exp + "\" + exp + ".pdf";

plot_graph(data, legends, lgnd_loc, x_tick, y_tick, name, x_label, y_label, axis_value, x_tick_lables, yscale_log, 1);


%% tr_size__lit_test_2d
exp = "tr_size__lit_test_2d";
path = project_dir() + "pde_2d\weights_results\burgers2d\" + exp + "\" + exp + ".csv";
data = load(path);

legends = {'train size-80', 'train size-100', 'train size-120'};
% legends = {'train size-30', 'train size-40', 'train size-50', 'train size-60', 'train size-70', 'train size-80', 'train size-90', 'train size-100', 'train size-110', 'train size-120'};
lgnd_loc = 'NW';
x_tick = 5:5:30; y_tick = 0; x_tick_lables = 0; yscale_log = 0;
axis_value = 0; x_label = 'Time index'; y_label = 'Prediction error';

name = project_dir() + "pde_2d\weights_results\burgers2d\" + exp + "\" + exp + ".pdf";

plot_graph(data, legends, lgnd_loc, x_tick, y_tick, name, x_label, y_label, axis_value, x_tick_lables, yscale_log, 1);

% path = project_dir() + "pde_2d\weights_results\burgers2d\" + exp + "\" + exp + "_9.csv";
% name = project_dir() + "pde_2d\weights_results\burgers2d\" + exp + "\" + exp + "_9.pdf";
% 
% plot_graph(data, legends, lgnd_loc, x_tick, y_tick, name, x_label, y_label, axis_value, x_tick_lables, yscale_log, 1);


%% B__dt_test_2d
a = load('C:\Users\yash kumar\PycharmProjects\gcn\src\pde_2d\Results\burgers2d\B__dt_test_2d\B__dt_test_2d.csv');
clr = {'y','m','c','r','g','b'};
figure;

t = 1:1:5;
y1 = a(1,:); c1 = ['-o', clr{1}];
y2 = a(2,:); c2 = ['-s', clr{2}];

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
set(gca,'LineWidth',2/2,'FontSize',28/2,'FontWeight','normal','FontName','Times')
set(get(gca,'xlabel'),'String','rollout time in testing ','FontSize',32/2','FontWeight',...
'bold','FontName','Times','Interpreter','tex')
set(get(gca,'ylabel'),'String','Prediction error','FontSize',32/2','FontWeight',...
'bold','FontName','Times','Interpreter','tex')
set(gcf,'Position',[1 1 round(1000/2) round(1000/2)])
h = legend({'Chebychev', 'None'},'location','SE','FontSize',28/2,'FontWeight','normal','FontName','Times');
% export_fig(gcf,'scatter1.eps','-eps','-r300')
export_fig(gcf,'B__dt_test_2d.pdf','-pdf','-r300')

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
a = load('C:\Users\yash kumar\PycharmProjects\gcn\src\pde_2d\Results\Delta{u}Delta{u}\PointNetLayer\No_Basis\NoBasis__del.csv');
clr = {'y','m','c','r','g','b'};
figure;

t = 5:4:35;%0:seconds(30):minutes(3);
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
h = legend({'GaussianRBF','None'},'location','NW','FontSize',28,'FontWeight','normal','FontName','Times');
% export_fig(gcf,'scatter1.eps','-eps','-r300')
export_fig(gcf,'NoBasis__del.pdf','-pdf','-r300')

% axis([.8 11 0.006 0.6])
% yticks([0.008 0.012 0.018 0.1 0.4])
xticks(t)
% xticklabels({'0','1','2','5','10'})

% axis([16 80 0,1.5])
% bb = linspace(0,1.5,15);
% yticks(bb);
% print -depsc PrMvsS
% export_fig(gcf,'PrOnlineNoise.pdf','-pdf','-r300')

%% time__epoch_2d
exp = "time__epoch_2d";
path1 = project_dir() + "pde_2d\weights_results\burgers2d\" + exp + "\" + exp + ".csv";
path2 = project_dir() + "pde_2d\weights_results\burgers2d\" + exp + "\" + exp + "_vry.csv";
a = load(path1); b = load(path2);
x_tick = 100:100:700; y_tick = 0; 
x_label = 'Epoch'; y_label_l = 'Prediction error'; y_label_r = 'Time elapsed (sec)';
legends = {'constant depth', 'depth refinement', 'constant depth', 'depth refinement'}; 
lgnd_loc = 'NW';
% a(2,:) = a(2,:)*100;
% b(2,:) = b(2,:)*100;
% ab = {a, b};
% clear abab
% abab(:, 1, :) = a.';
% abab(:, 2, :) = b.';
% cc = {1:20};
% plotBarStackGroups(abab, cc);
% ==========================
% abb = [a(1, :); b(1, :); a(2, :)*100; b(2, :)*100].';
% bar(abb)

yyaxis left
plot(x_tick, a(2, :), '-o', 'MarkerSize',12, 'LineWidth',2);
hold on
plot(x_tick, b(2, :), '-s', 'MarkerSize',12, 'LineWidth',2);
set(get(gca,'ylabel'),'String',y_label_l,'FontSize',32/2','FontWeight',...
'bold','FontName','Times','Interpreter','tex')
h = legend(legends,'location',lgnd_loc,'FontSize',28/3,'FontWeight','normal','FontName','Times','Orientation','Vertical');

yyaxis right
plot(x_tick, a(1, :), '-o', 'MarkerSize',12, 'LineWidth',2);
hold on
plot(x_tick, b(1, :), '-s', 'MarkerSize',12, 'LineWidth',2);
set(get(gca,'ylabel'),'String',y_label_r,'FontSize',32/2','FontWeight',...
'bold','FontName','Times','Interpreter','tex')

set(gcf,'color','w')
set(gca,'LineWidth',2/2,'FontSize',28/2,'FontWeight','normal','FontName','Times')
set(get(gca,'xlabel'),'String', x_label,'FontSize',32/2','FontWeight',...
'bold','FontName','Times','Interpreter','tex')
set(gcf,'Position',[1 1 round(1000/2) round(1000/2)])

h = legend(legends,'location', lgnd_loc, 'FontSize',28/3,'FontWeight','normal','FontName','Times','Orientation','Vertical');

name = project_dir() + "pde_2d\weights_results\burgers2d\" + exp + "\" + exp + ".pdf";
export_fig(gcf, name, '-pdf', '-r300')

%% B__bt_test_2d
a = load('C:\Users\yash kumar\PycharmProjects\gcn\src\pde_2d\Results\burgers2d\B__bt_test_2d\B__bt_test_2d.csv');
clr = {'y','m','c','r','g','b'};
figure;

t = 2:2:14;
y1 = a(1,:); c1 = ['-o', clr{1}];
y2 = a(2,:); c2 = ['-s', clr{2}];
y3 = a(3,:); c3 = ['-*', clr{3}];
y4 = a(4,:); c4 = ['-p', clr{4}];
y5 = a(5,:); c5 = ['-+', clr{5}];
y6 = a(6,:); c6 = ['-d', clr{6}];

plot(...
    t,y1,c1,... 
    t,y2,c2,... 
    t,y3,c3,... 
    t,y4,c4,... 
    t,y5,c5,...
    t,y6,c6,... 
    'MarkerSize',12,...
    'LineWidth',2);
% p(1).MarkerEdgeColor = 'green';
% plot(t,y,'DurationTickFormat','mm:ss','-o')
% tl = ['no. of basis =', int2str(n_B(i))]
% title(tl)


set(gcf,'color','w')
% set(gca, 'YScale', 'log')
set(gca,'LineWidth',2/2,'FontSize',28/2,'FontWeight','normal','FontName','Times')
set(get(gca,'xlabel'),'String','rollout time in testing ',...
    'FontSize',32/2','FontWeight','bold','FontName','Times',...
    'Interpreter','tex')
set(get(gca,'ylabel'),'String','Prediction error','FontSize',...
    32/2','FontWeight','bold','FontName','Times','Interpreter','tex')
set(gcf,'Position',[1 1 round(1000/2) round(1000/2)])
h = legend({'Chebychev', 'Fourier', 'VanillaRBF',...
    'GaussianRBF', 'MultiquadRBF', 'None'},'location','SE',...
    'FontSize',28/2,'FontWeight','normal','FontName','Times');
% export_fig(gcf,'scatter1.eps','-eps','-r300')
export_fig(gcf,'B__bt_test_2d.pdf','-pdf','-r300')

% axis([.8 11 0.006 0.6])
% yticks([0.008 0.012 0.018 0.1 0.4])
xticks(t)
% xticklabels({'0','1','2','5','10'})

% axis([16 80 0,1.5])
% bb = linspace(0,1.5,15);
% yticks(bb);
% print -depsc PrMvsS
% export_fig(gcf,'PrOnlineNoise.pdf','-pdf','-r300')

%% B__n_B_2d
a = load('C:\Users\yash kumar\PycharmProjects\gcn\src\pde_2d\Results\burgers2d\B__n_B_2d\B__n_B_2d.csv');
clr = {'y','m','c','r','g','b'};
figure;

t = [1, 2, 3, 5, 10];%0:seconds(30):minutes(3);
y = rand(1,5);
y1 = a(1,:); c1 = ['-o', clr{1}];
y2 = a(2,:); c2 = ['-s', clr{2}];
y3 = a(3,:); c3 = ['-*', clr{3}];
y4 = a(4,:); c4 = ['-p', clr{4}];
y5 = a(5,:); c5 = ['-+', clr{5}];
y6 = a(6,:); c6 = ['-d', clr{6}];

plot(...
    t,y1,c1,... 
    t,y2,c2,... 
    t,y3,c3,... 
    t,y4,c4,... 
    t,y5,c5,...
    t,y6,c6,... 
    'MarkerSize',12,...
    'LineWidth',2);
% p(1).MarkerEdgeColor = 'green';
% plot(t,y,'DurationTickFormat','mm:ss','-o')
% tl = ['no. of basis =', int2str(n_B(i))]
% title(tl)


set(gcf,'color','w')
% set(gca, 'YScale', 'log')
set(gca,'LineWidth',2,'FontSize',28,'FontWeight','normal','FontName','Times')
set(get(gca,'xlabel'),'String','Number of basis fn','FontSize',32','FontWeight',...
'bold','FontName','Times','Interpreter','tex')
set(get(gca,'ylabel'),'String','Prediction error','FontSize',32','FontWeight',...
'bold','FontName','Times','Interpreter','tex')
set(gcf,'Position',[1 1 round(1000) round(1000)])
h = legend({'Chebychev', 'Fourier', 'VanillaRBF', 'GaussianRBF', 'MultiquadRBF', 'Polynomial'},'location','NE','FontSize',28,'FontWeight','normal','FontName','Times');
% export_fig(gcf,'scatter1.eps','-eps','-r300')
export_fig(gcf,'B__n_B.pdf','-pdf','-r300')

% axis([.8 11 0.006 0.6])
% yticks([0.008 0.012 0.018 0.1 0.4])
xticks([1, 2, 3, 5, 10])
% xticklabels({'0','1','2','5','10'})

% axis([16 80 0,1.5])
% bb = linspace(0,1.5,15);
% yticks(bb);
% print -depsc PrMvsS
% export_fig(gcf,'PrOnlineNoise.pdf','-pdf','-r300')

%%  lit_train__lit_test_2d

exp = "lit_train__lit_test_2d";
path = project_dir() + "pde_2d\weights_results\burgers2d\" + exp + "\" + exp + ".csv";
data = load(path);

legends = {'lit train-2', 'lit train-3', 'lit train-4'};
lgnd_loc = 'NW';
x_tick = 5:5:30; y_tick = 0; x_tick_lables = 0; yscale_log = 0;
x_label = 'Time index'; y_label = 'Prediction error';

name = project_dir() + "pde_2d\weights_results\burgers2d\" + exp + "\" + exp + ".pdf";

plot_graph(data, legends, lgnd_loc, x_tick, y_tick, name, x_label, y_label, axis_value, x_tick_lables, yscale_log, 1);

%%
% a = load('C:\Users\yash kumar\PycharmProjects\gcn\src\PDEs\Results\Delta{u}Delta{u}\PointNetLayer\time__epoch.csv');
% b = load('C:\Users\yash kumar\PycharmProjects\gcn\src\PDEs\Results\Delta{u}Delta{u}\PointNetLayer\time__epoch1.csv');
% ab = {a, b};
% 
% NumGroupsPerAxis = 2;%size(stackData, 1);
% NumStacksPerGroup = 20;%size(stackData, 2);
% % Count off the number of bins
% groupBins = 1:NumGroupsPerAxis;
% MaxGroupWidth = 0.65; % Fraction of 1. If 1, then we have all bars in groups touching
% groupOffset = MaxGroupWidth/NumStacksPerGroup;
% figure;
% hold on;
% t = 1:20;
% for j=1:2
% % aa = [a(1,:); a(2,:)];
% internalPosCount = i - ((NumStacksPerGroup+1) / 2);
% groupDrawPos = (internalPosCount)* groupOffset + groupBins;
% b(j,:) = bar(t, ab{1,j}, 'stacked');
% set(b(j,:),'BarWidth',groupOffset);
% set(b(j,:),'XData',groupDrawPos);
% end
% hold off;
% 

%%

function project_dir = project_dir()
project_dir = "G:\My Drive\Colab Notebooks\gnode_pde\src\";
end


function plot_graph(data, legends, lgnd_loc, x_tick, y_tick, name, x_label, y_label, axis_value, x_tick_lables, yscale_log, m_legend)

disp(data);
clr = {'b','m','g','r','c','k','y',[0.75, 0, 0.75], [0.6350, 0.0780, 0.1840], [0.6350, 0.0780, 0.1840], [0.6350, 0.0780, 0.1840], [0.6350, 0.0780, 0.1840]};
shapes = {'-o', '-s', '-*', '-p', '-+', '-d', '-<', '->', '-^', '-h', '--'};
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
%     a=axes('position', get(gca,'position'),'visible','off');
%     h2 = legend(a, p(4:6), legends(4:6),'FontWeight','normal','FontName','Times','Orientation','Horizontal');
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


