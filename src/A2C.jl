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


name = "A2C"

save_dir = make_save_dir(name)

lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
rng = MersenneTwister(123)

N_ENV = 16
UPDATE_FREQ = 10
env = MultiThreadEnv([
    LunarLander() for i in 1:N_ENV
])

ns, na = length(get_state(env[1])), length(get_actions(env[1]))




# RLBase.reset!(env, is_force = true)
agent = Agent(
    policy = QBasedPolicy(
        learner = A2CLearner(
            approximator = ActorCritic(
                actor = small_net_model(ns, na),
                critic = small_net_model(ns, 1),
                optimizer = ADAM(),
            ) |> cpu,
            γ = 0.99f0,
            actor_loss_weight = 1.0f0,
            critic_loss_weight = 0.5f0,
            entropy_loss_weight = 0.001f0,
        ),
        explorer = BatchExplorer(GumbelSoftmaxExplorer()),#= seed = nothing =#
    ),
    trajectory = CircularCompactSARTSATrajectory(;
        capacity = UPDATE_FREQ,
        state_type = Float32,
        state_size = (ns, N_ENV),
        action_type = Int,
        action_size = (N_ENV,),
        reward_type = Float32,
        reward_size = (N_ENV,),
        terminal_type = Bool,
        terminal_size = (N_ENV,),
    ),
)
stop_condition = StopAfterStep(Conf.duration)
total_reward_per_episode = TotalBatchRewardPerEpisode(N_ENV)
time_per_step = TimePerStep()
hook = ComposedHook(
    total_reward_per_episode,
    time_per_step,
    DoEveryNStep() do t, agent, env
        with_logger(lg) do
            @info(
                "training_AC",
                actor_loss = agent.policy.learner.actor_loss,
                critic_loss = agent.policy.learner.critic_loss,
                entropy_loss = agent.policy.learner.entropy_loss,
                loss = agent.policy.learner.loss,
            )
            for i in 1:length(env)
                if get_terminal(env[i])
                    @info "training_AC" reward = total_reward_per_episode.rewards[i][end] log_step_increment =
                        0
                    break
                end
            end
        end
    end,
    DoEveryNStep(Conf.save_freq) do t, agent, env
        RLCore.save(save_dir, agent)
        BSON.@save joinpath(save_dir, "stats.bson") total_reward_per_episode time_per_step
    end,
)

infos = @timeRet run(agent, env, stop_condition, hook)
open(f->write(f, infos), joinpath(save_dir, "performance.txt"), "w")