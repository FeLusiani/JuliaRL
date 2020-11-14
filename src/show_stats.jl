#JuliaRL/src/show_stats.jl

using BSON
using PyPlot


"""
    show_stats(save_dir)

Plots the stats saved in the `stats.bson` file inside of `save_dir`.
Stats are total reward per episode, and time per step.
"""
function show_stats(save_dir::T) where {T<:AbstractString}
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
end
