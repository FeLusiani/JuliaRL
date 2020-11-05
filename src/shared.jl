using ReinforcementLearning
using Flux
using Dates

function LunarLander()
    inner_env = RLEnvs.GymEnv("LunarLander-v2")
    env = inner_env |> ActionTransformedEnv(a -> a-1;mapping= a -> a+1)
end

function make_save_dir(name::T) where {T<:AbstractString}
    t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
    save_dir = joinpath(pwd(), "checkpoints", "$name_$t")
    
    if isdir(save_dir)
        rm(save_dir; force=true, recursive=true)
    end

    save_dir
end


function net_model(ns::Int, na::Int)
    Chain(
        Dense(ns, 64, relu; initW = glorot_uniform(rng)),
        Dense(64, 64, relu; initW = glorot_uniform(rng)),
        Dense(64, 32, relu; initW = glorot_uniform(rng)),
        Dense(32, na; initW = glorot_uniform(rng)),
    ) |> cpu
end
