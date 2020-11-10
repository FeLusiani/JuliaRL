module Conf

# BasicDQN
duration = 150_000
batch_size = 64
min_replay_history = 100#div(duration,500) #100#400
decay_steps = 3000#div(duration,40) #3000#2000
capacity = div(duration,1)
save_freq = div(duration,2)

# 1000-400 no

# DQN
update_freq = 4
target_update_freq = 100
# RLmodel = "BasicDQN"
# save_dir = "/home/ubuntu/JuliaRL/checkpoints/"

end

# ENV["save_dir"]