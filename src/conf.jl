#JuliaRL/src/conf.jl

# Configuration values for the DQN algorithms are defined here
module Conf

# BasicDQN, DQN, PDQN

# duration in steps of the training
duration = 150_000
# size of mini-batches
batch_size = 64
# number of transitions that should be experienced before updating the approximator
min_replay_history = 100
# decay_steps for EpsilonGreedyExplorer
decay_steps = 3000
# capacity (in steps) of the experience buffer
capacity = duration
# frequency at which the agent is saved during training
save_freq = div(duration,2)

# DQN, PDQN

# the frequency of updating the approximator
update_freq = 4
# the frequency of updating the target
target_update_freq = 100

end