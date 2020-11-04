using ReinforcementLearning
using PyCall
using ReinforcementLearningEnvironments
using Random
using Flux
using TensorBoardLogger
using Dates
using Logging
using BSON

duration = 100_000

save_dir = nothing

description = """
This experiment uses three dense layers to approximate the Q value.
The testing environment is LunarLander-v2.
"""

if isnothing(save_dir)
    t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
    save_dir = joinpath(pwd(), "checkpoints", "BasicDQN_Lander")
end

if isdir(save_dir)
    rm(save_dir; force=true, recursive=true)
end

lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
rng = MersenneTwister(123)

inner_env = RLEnvs.GymEnv("LunarLander-v2")

env = inner_env |> ActionTransformedEnv(a -> a-1)
RLBase.get_actions(env::typeof(env)) = 1:4

ns, na = length(get_state(env)), length(get_actions(env))

agent = Agent(
    policy = QBasedPolicy(
        learner = BasicDQNLearner(
            approximator = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, 128, relu; initW = glorot_uniform(rng)),
                    Dense(128, 128, relu; initW = glorot_uniform(rng)),
                    Dense(128, na; initW = glorot_uniform(rng)),
                ) |> cpu,
                optimizer = ADAM(),
            ),
            batch_size = 32,
            min_replay_history = 100,
            loss_func = huber_loss,
            rng = rng,
        ),
        explorer = EpsilonGreedyExplorer(
            kind = :exp,
            ϵ_stable = 0.01,
            decay_steps = 500,
            rng = rng,
        ),
    ),
    trajectory = CircularCompactSARTSATrajectory(
        capacity = 1000,
        state_type = Float32,
        state_size = (ns,),
    ),
)

stop_condition = StopAfterStep(duration)

total_reward_per_episode = TotalRewardPerEpisode()
time_per_step = TimePerStep()
hook = ComposedHook(
    total_reward_per_episode,
    time_per_step,
    DoEveryNStep() do t, agent, env
        with_logger(lg) do
            @info "training" loss = agent.policy.learner.loss
        end
    end,
    DoEveryNEpisode() do t, agent, env
        with_logger(lg) do
            @info "training" reward = total_reward_per_episode.rewards[end] log_step_increment =
                0
        end
    end,
    DoEveryNStep(duration) do t, agent, env
        RLCore.save(save_dir, agent)
        BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
    end,
)


run(agent, env, stop_condition, hook)


