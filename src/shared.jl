using ReinforcementLearning
using Flux
using Dates
using Suppressor

function LunarLander()
    inner_env = RLEnvs.GymEnv("LunarLander-v2")
    env = inner_env |> ActionTransformedEnv(a -> a-1;mapping= a -> a+1)
end

function make_save_dir(name::T) where {T<:AbstractString}
    # t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
    save_dir = joinpath(pwd(), "checkpoints", "$name")
    
    if isdir(save_dir)
        rm(save_dir; force=true, recursive=true)
    end

    save_dir
end


function net_model(ns::Int, na::Int)
    Chain(
        Dense(ns, 64, leakyrelu; initW = glorot_uniform(rng)),
        Dense(64, 64, leakyrelu; initW = glorot_uniform(rng)),
        Dense(64, 32, leakyrelu; initW = glorot_uniform(rng)),
        Dense(32, na; initW = glorot_uniform(rng)),
    ) |> cpu
end

function small_net_model(ns::Int, na::Int)
    Chain(
        Dense(ns, 32, leakyrelu; initW = glorot_uniform(rng)),
        Dense(32, 32, leakyrelu; initW = glorot_uniform(rng)),
        Dense(32, 16, leakyrelu; initW = glorot_uniform(rng)),
        Dense(16, na; initW = glorot_uniform(rng)),
    ) |> cpu
end


macro timeRet(ex)
    quote
        while false; end # compiler heuristic: compile this block (alter this if the heuristic changes)
        local stats = Base.gc_num()
        local elapsedtime = time_ns()
        local val = $(esc(ex))
        elapsedtime = Base.time_ns() - elapsedtime
        local diff = Base.GC_Diff(Base.gc_num(), stats)
        local output = @capture_out Base.time_print(
            elapsedtime, diff.allocd,
            diff.total_time,
            Base.gc_alloc_count(diff)
        )
        print(output)
        println()
        output
    end
end