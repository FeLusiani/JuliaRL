#JuliaRL/src/display.jl

using ReinforcementLearning
using PyCall
using Random
using ReinforcementLearningEnvironments
using Flux
using Logging

include("./shared.jl")

"""
    display_agent(save_dir, duration::Int = 1000)

Runs the agent saved at `save_dir`, rendering the environment,
up to `duration` steps.
"""
function display_agent(save_dir, duration::Int = 1000)
    env = LunarLander()
    agent = RLCore.load(save_dir, Agent)
    Flux.testmode!(agent)

    stop_condition = StopAfterStep(duration)
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
end