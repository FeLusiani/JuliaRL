using ReinforcementLearning
using Flux
using Dates
using Suppressor

function pyenv2env(pyenv::PyObject)
    obs_space = convert(AbstractSpace, pyenv.observation_space)
    act_space = convert(AbstractSpace, pyenv.action_space)
    obs_type = if obs_space isa Union{MultiContinuousSpace,MultiDiscreteSpace}
        PyArray
    elseif obs_space isa ContinuousSpace
        Float64
    elseif obs_space isa DiscreteSpace
        Int
    elseif obs_space isa VectSpace
        PyVector
    elseif obs_space isa DictSpace
        PyDict
    else
        error("don't know how to get the observation type from observation space of $obs_space")
    end
    env = GymEnv{obs_type,typeof(act_space),typeof(obs_space),typeof(pyenv)}(
        pyenv,
        obs_space,
        act_space,
        PyNULL(),
    )
    reset!(env) # reset immediately to init env.state
    env
end



function LunarLander(;particles=false)
    gym = pyimport("CustomGym")
    gym = pyimport("importlib").reload(gym.lunar_lander)
    gym.LunarLander(particles) |>
    pyenv2env |>
    ActionTransformedEnv(a -> a-1;mapping= a -> a+1)
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

function shallow_net_model(ns::Int, na::Int)
    Chain(
        Dense(ns, 128, leakyrelu; initW = glorot_uniform(rng)),
        Dense(128, 128, leakyrelu; initW = glorot_uniform(rng)),
        Dense(128, 64, leakyrelu; initW = glorot_uniform(rng)),
        Dense(64, na; initW = glorot_uniform(rng)),
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

