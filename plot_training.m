clear; clc; close all;

% Load saved training data
load("trainingStats_final.mat","trainingStats");

figure;

% Episode reward
plot(trainingStats.EpisodeIndex,trainingStats.EpisodeReward,'Color',[0.2 0.6 1]); hold on;

% Average reward
plot(trainingStats.EpisodeIndex,trainingStats.AverageReward,'LineWidth',2);

%legend('Episode Reward','Average Reward','Location','best')
xlabel('Episode');
ylabel('Value');
title('Training Progress');


set(gcf,'Color','none');   % figure background
set(gca,'Color','none');   % axes background

exportgraphics(gcf,'plot.png',...
    'BackgroundColor','none',...
    'Resolution',300);

% Q0 (if exists)
%if isfield(trainingStats,'EpisodeQ0')
%%
figure;
    plot(trainingStats.EpisodeQ0,'Color',[1 0.6 0]);
    %legend('Episode Reward','Average Reward','Q0','Location','best');

legend('Q0','Location','best')
xlabel('Episode');
ylabel(' Episode Q0 Value');
title('Training Progress');
grid on;
%%
% Save high-quality image
%exportgraphics(gcf,'training_plot.png','Resolution',300);

% Save MATLAB figure
%savefig(gcf,'training_plot.fig');
