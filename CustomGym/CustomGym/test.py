from CustomGym.lunar_lander import LunarLander, heuristic
import gym
from timeit import Timer

N_STEPS = 200000
env = LunarLander()


def run_test():
    steps = 0
    while steps <= N_STEPS:
        s = env.reset()
        # play episode until it ends or we reach N_STEPS
        while True:
            a = heuristic(env, s)
            s, r, done, info = env.step(a)

            steps += 1
            if done: break
            if steps > N_STEPS: break


setup = """
gc.enable()
from __main__ import run_test
"""
t1 = Timer('run_test()', setup).timeit(number=1)
print(f"Environment:\tCustomGym\nN_STEPS:\t{N_STEPS}\nTime:\t\t{t1}")

env = gym.make("LunarLander-v2")
t2 = Timer('run_test()', setup).timeit(number=1)
print(f"Environment:\tGym\nN_STEPS:\t{N_STEPS}\nTime:\t\t{t2}")

print("")
print(f"t2 / t1 ratio:\t{t2 / t1}")