using ReinforcementLearning
using PyCall
using ReinforcementLearningEnvironments
using Random
using Flux
using TensorBoardLogger
using Dates
using Logging
using BSON
include("./conf.jl")
include("./shared.jl")

duration = 100_000
name = "DQN"

save_dir = make_save_dir(name)

lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
rng = MersenneTwister(123)

inner_env = RLEnvs.GymEnv("LunarLander-v2")
env = inner_env |> ActionTransformedEnv(a -> a-1)
RLBase.get_actions(::typeof(env)) = 1:4
ns, na = length(get_state(env)), length(get_actions(env))

agent = Agent(
        policy = QBasedPolicy(
            learner = DQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = net_model(ns, na),
                    optimizer = ADAM(),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = net_model(ns, na),
                ),
                loss_func = huber_loss,
                stack_size = nothing,
                batch_size = 64,
                update_horizon = 1,
                min_replay_history = 100,
                update_freq = 4,
                target_update_freq = 100,
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 50_000,
                rng = rng,
            ),
        ),
        trajectory = CircularCompactSARTSATrajectory(
            capacity = 100_000,
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
        if agent.policy.learner.update_step % agent.policy.learner.update_freq == 0
            with_logger(lg) do
                @info "training" loss = agent.policy.learner.loss
            end
        end
    end,
    DoEveryNEpisode() do t, agent, env
        with_logger(lg) do
            @info "training" reward = total_reward_per_episode.rewards[end]
            log_step_increment = 0
        end
    end,
    DoEveryNStep(div(duration,5)) do t, agent, env
        RLCore.save(save_dir, agent)
        BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
    end,
)


run(agent, env, stop_condition, hook)


