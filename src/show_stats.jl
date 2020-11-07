using BSON
using PyPlot

include("conf.jl")

if !@isdefined(save_dir)
    @error "You must set save_dir variable before including this script."
else
    @info "save_dir = $save_dir"
end

BSON.@load joinpath(save_dir,"stats.bson") total_reward_per_episode time_per_step

figure(figsize=(10, 5))
subplot(121)
title(save_dir)
ylabel("Total reward")
xlabel("Episode")
plot(total_reward_per_episode.rewards)

subplot(122)
ylabel("Time")
xlabel("Step")
x = (1:length(time_per_step.times)) * 100
y = time_per_step.times / 100
plot(x, time_per_step.times)
