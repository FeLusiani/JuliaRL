using BSON
using PyPlot

include("conf.jl")

BSON.@load joinpath(ENV["save_dir"],"stats.bson") total_reward_per_episode time_per_step

figure(figsize=(10, 5))
subplot(121)
ylabel("Total reward")
xlabel("Episode")
plot(total_reward_per_episode.rewards)
subplot(122)
ylabel("Time")
xlabel("Episode")
plot(time_per_step.times .* 100)
