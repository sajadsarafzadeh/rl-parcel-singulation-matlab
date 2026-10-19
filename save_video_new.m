clear; clc; close all;

load("agent_trained_final.mat","agent");

env = RL_environment();
obs = reset(env);

plot(env);
drawnow;
pause(0.5);

% Find the new simple visualizer figure
set(0,'ShowHiddenHandles','on');
fig = findall(0,'Type','figure','Name','AMS sorting visualizer simple');
set(0,'ShowHiddenHandles','off');

if isempty(fig)
    error('AMS simple visualizer figure not found.');
end

fig = fig(1);

videoObj = VideoWriter('trained_agent_video_simple.mp4','MPEG-4');
videoObj.FrameRate = round(1/env.dt);   % dt=0.01 -> 100 fps
open(videoObj);

isDone = false;
maxSteps = 1000;

for k = 1:maxSteps

    actionCell = getAction(agent, obs);
    action = actionCell{1};

    [obs, reward, isDone, Info] = step(env, action);

    drawnow;

    frame = getframe(fig);   % capture whole simple visualizer window
    writeVideo(videoObj, frame);

    if isDone
        break;
    end
end

close(videoObj);
disp('Video saved successfully.');
