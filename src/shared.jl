#JuliaRL/src/shared.jl

using ReinforcementLearning
using ReinforcementLearningEnvironments
using Flux
using Dates
using Suppressor


"""
    pyenv2env(pyenv::PyObject)

Returns a `ReinforcementLearningEnvironments.GymEnv` environment
from the `gym` environment `pyenv`.
Code is from `ReinforcementLearningEnvironments.GymEnv(::String)` function.
"""
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
    RLBase.reset!(env) # reset immediately to init env.state
    env
end



"""
Returns a `LunarLander` environment from the `CustomGym` python module
as a `ReinforcementLearningEnvironments.GymEnv`
"""
function LunarLander()
    gym = pyimport("CustomGym")
    # reload the python module, otherwise changes to the code
    # won't be effective until you restart julia
    gym = pyimport("importlib").reload(gym.lunar_lander)
    gym.LunarLander() |>
    pyenv2env |>
    ActionTransformedEnv(a -> a-1;mapping= a -> a+1)

end


function net_model(ns::Int, na::Int, rng)
    Chain(
        Dense(ns, 64, leakyrelu; initW = glorot_uniform(rng)),
        Dense(64, 64, leakyrelu; initW = glorot_uniform(rng)),
        Dense(64, 32, leakyrelu; initW = glorot_uniform(rng)),
        Dense(32, na; initW = glorot_uniform(rng)),
    ) |> cpu
end


function shallow_net_model(ns::Int, na::Int, rng)
    Chain(
        Dense(ns, 128, leakyrelu; initW = glorot_uniform(rng)),
        Dense(128, 64, leakyrelu; initW = glorot_uniform(rng)),
        Dense(64, na; initW = glorot_uniform(rng)),
    ) |> cpu
end


"""
@timeRet expr

Works like the macro `@time` from `Base`,
but returns the printed statistics as a string. 
"""
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


"""
    log_training_info(infos, agent, save_dir)

Creates file `training_infos.txt` inside of the `save_dir` directory,
containing training infos (agent structure and elapsed time)
"""
function log_training_info(infos, agent, save_dir)
    file_path = joinpath(save_dir, "training_infos.txt")
    open(f->write(f, infos*"\n", string(agent)), file_path, "w")
end