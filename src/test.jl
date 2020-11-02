using ReinforcementLearning
using PyCall
using ReinforcementLearningEnvironments
using Random
using Flux

# include("./gym.jl")

rng = MersenneTwister(123)
inner_env = RLEnvs.GymEnv("LunarLander-v2")
env = inner_env |> ActionTransformedEnv(a -> a-1)
RLBase.get_actions(env::typeof(env)) = 1:4

ns = length((get_state(env)))
na = length(get_actions(env))

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


experiment     = E`JuliaRL_BasicDQN_CartPole`
# agent          = experiment.agent
# env            = experiment.env
stop_condition = StopAfterStep(100_000)
hook           = TotalRewardPerEpisode()

run(agent, env, stop_condition, hook)