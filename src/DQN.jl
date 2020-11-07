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


name = "DQN"

save_dir = make_save_dir(name)

lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
rng = MersenneTwister(123)

env = LunarLander()
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
                batch_size = Conf.batch_size,
                update_horizon = 1,
                min_replay_history = Conf.min_replay_history,
                update_freq = Conf.update_freq,
                target_update_freq = Conf.target_update_freq,
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = Conf.decay_steps,
                rng = rng,
            ),
        ),
        trajectory = CircularCompactSARTSATrajectory(
            capacity = Conf.capacity,
            state_type = Float32,
            state_size = (ns,),
        ),
)

stop_condition = StopAfterStep(Conf.duration)
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
            # log_step_increment = 0
        end
    end,
    DoEveryNStep(Conf.save_freq) do t, agent, env
        RLCore.save(save_dir, agent)
        BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
    end,
)


@time run(agent, env, stop_condition, hook)


