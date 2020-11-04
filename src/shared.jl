using ReinforcementLearning
using Flux

function make_save_dir(name::T) where {T<:AbstractString}
    save_dir = joinpath(pwd(), "checkpoints", "$name")
    # t = Dates.format(now(), "HH_MM_SS")
    
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
