using ReinforcementLearning
using PyCall
using Random
using ReinforcementLearningEnvironments
using Flux

include("./conf.jl")
include("./shared.jl")

env = LunarLander()

agent = RLCore.load(Conf.save_dir, Agent)

Flux.testmode!(agent)

stop_condition = StopAfterStep(Conf.duration)
disp_hook = DoEveryNStep(1) do t, agent, env
    env.env.pyenv.render()
end

run(agent, env, stop_condition, disp_hook)