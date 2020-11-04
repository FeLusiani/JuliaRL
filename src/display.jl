using ReinforcementLearning
using PyCall
using Random
using ReinforcementLearningEnvironments
using Flux

include("./conf.jl")

inner_env = RLEnvs.GymEnv("LunarLander-v2")
env = inner_env |> ActionTransformedEnv(a -> a-1)
RLBase.get_actions(::typeof(env)) where {T} = 1:4

println("Experiment:")
experiment = readline()
agent = RLCore.load("/home/ubuntu/JuliaRL/checkpoints/$experiment", Agent)

Flux.testmode!(agent)

stop_condition = StopAfterStep(10_000)
disp_hook = DoEveryNStep(1) do t, agent, env
    env.env.pyenv.render()
end

run(agent, env, stop_condition, disp_hook)