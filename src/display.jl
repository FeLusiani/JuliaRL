using ReinforcementLearning
using Flux

inner_env = RLEnvs.GymEnv("LunarLander-v2")
env = inner_env |> ActionTransformedEnv(a -> a-1)
RLBase.get_actions(env::typeof(env)) = 1:4

agent = RLCore.load("/home/ubuntu/JuliaRL/checkpoints/JuliaRL_BasicDQN_CartPole_2020_11_02_23_16_11", Agent)

Flux.testmode!(agent)

stop_condition = StopAfterStep(10_000)
disp_hook = DoEveryNStep(1) do t, agent, env
    env.env.pyenv.render()
    end

run(agent, env, stop_condition, disp_hook)