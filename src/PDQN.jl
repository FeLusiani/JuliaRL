#JuliaRL/src/PDQN.jl

using ReinforcementLearning
using PyCall
using ReinforcementLearningEnvironments
using Random
using Flux
using TensorBoardLogger
using Dates
using Logging
using BSON
using Suppressor
include("./conf.jl")
include("./shared.jl")


"""
    runPDQN(save_dir)

Train an agent using PDQN learning.
Results will be saved at `save_dir`.
"""
function runPDQN(save_dir::T) where {T<:AbstractString}
    # clear save_dir directory
    isdir(save_dir) && rm(save_dir; force=true, recursive=true)

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = MersenneTwister(123)

    env = LunarLander()
    ns, na = length(get_state(env)), length(get_actions(env))

    agent = Agent(
        policy = QBasedPolicy(
            learner = PrioritizedDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = net_model(ns, na, rng),
                    optimizer = ADAM(),
                ),
                target_approximator = NeuralNetworkApproximator(
                    model = net_model(ns, na, rng),
                    optimizer = ADAM(),
                ),
                loss_func = huber_loss_unreduced,
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
        trajectory = CircularCompactPSARTSATrajectory(
            capacity = Conf.capacity,
            state_type = Float32,
            state_size = (ns,),
        ),
    )

    global episode_loss = 0
    stop_condition = StopAfterStep(Conf.duration)
    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        DoEveryNStep() do t, agent, env
            if agent.policy.learner.update_step % agent.policy.learner.update_freq == 0
                global episode_loss += agent.policy.learner.loss
            end
        end,
        DoEveryNEpisode() do t, agent, env
            with_logger(lg) do
                global episode_loss
                @info "training" loss = episode_loss
                @info "training" reward = total_reward_per_episode.rewards[end]
                log_step_increment = 0
                episode_loss = 0
            end
            episode_loss = 0
        end,
        DoEveryNStep(Conf.save_freq) do t, agent, env
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )


    infos = @timeRet run(agent, env, stop_condition, hook)
    log_training_info(infos, agent, save_dir)
end