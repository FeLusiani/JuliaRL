using ReinforcementLearning
using PyCall
using Random
using ReinforcementLearningEnvironments
using Flux
using Logging

include("./conf.jl")
include("./shared.jl")

if !@isdefined(save_dir)
    @error "You must set save_dir variable before including this script."
else
    @info "save_dir = $save_dir"
end
    

env = LunarLander()

agent = RLCore.load(save_dir, Agent)

Flux.testmode!(agent)

stop_condition = StopAfterStep(Conf.duration)
disp_hook = DoEveryNStep(1) do t, agent, env
    env.env.pyenv.render()
end

disp_hook = DoEveryNStep(1) do t, agent, env
    env.env.pyenv.render()
end

print_hook = DoEveryNEpisode(1) do t, agent, env
    println("- Ep N $t")
end

hook = ComposedHook(disp_hook, print_hook)

run(agent, env, stop_condition, hook)