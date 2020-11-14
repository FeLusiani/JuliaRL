#JuliaRL/src/BasicDQN.jl

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
    runBasicDQN(save_dir)

Train an agent using BasicDQN learning.
Results will be saved at `save_dir`.
"""
function runBasicDQN(save_dir::T) where {T<:AbstractString}
    # clear save_dir directory
    isdir(save_dir) && rm(save_dir; force=true, recursive=true)

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = MersenneTwister(123)

    env = LunarLander()

    ns, na = length(get_state(env)), length(get_actions(env))

    agent = Agent(
        policy = QBasedPolicy(
            learner = BasicDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = net_model(ns, na, rng),
                    optimizer = ADAM(),
                ),
                batch_size = Conf.batch_size,
                min_replay_history = Conf.min_replay_history,
                loss_func = huber_loss,
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

    global episode_loss = 0
    total_reward_per_episode = TotalRewardPerEpisode()
    time_per_step = TimePerStep()
    hook = ComposedHook(
        total_reward_per_episode,
        time_per_step,
        DoEveryNStep() do t, agent, env
            global episode_loss += loss = agent.policy.learner.loss
        end,
        DoEveryNEpisode() do t, agent, env
            with_logger(lg) do
                global episode_loss
                @info "training" loss = episode_loss
                @info "training" reward = total_reward_per_episode.rewards[end]
                log_step_increment = 0
                episode_loss = 0
            end
        end,
        DoEveryNStep(Conf.save_freq) do t, agent, env
            RLCore.save(save_dir, agent)
            BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
        end,
    )


    infos = @timeRet run(agent, env, stop_condition, hook)
    log_training_info(infos, agent, save_dir)
end




