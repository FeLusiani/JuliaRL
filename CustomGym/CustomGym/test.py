from .lunar_lander import LunarLander, heuristic

N_STEPS = 10000
env = LunarLander()
steps = 0

while steps <= N_STEPS:
    while True:
        env.reset()
        a = heuristic(env, s)
        s, r, done, info = env.step(a)

        steps += 1
        if done: break
        if steps > N_STEPS: break
