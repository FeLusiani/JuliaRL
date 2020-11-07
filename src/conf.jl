module Conf

# BasicDQN
duration = 100_000
batch_size = 32
min_replay_history = 100
decay_steps = div(duration,10)
capacity = div(duration,10)
save_freq = div(duration,1)

# DQN
update_freq = 4
target_update_freq = 100
# RLmodel = "BasicDQN"
# save_dir = "/home/ubuntu/JuliaRL/checkpoints/"

end

# ENV["save_dir"]