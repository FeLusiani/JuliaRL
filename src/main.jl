# using Pkg

# Pkg.add(PackageSpec(url="https://github.com/JuliaML/OpenAIGym.jl.git"))

using OpenAIGym
env = GymEnv(:BipedalWalker, :v3)
for i ∈ 1:20
  T = 0
  R = run_episode(env, RandomPolicy()) do (s, a, r, s′)
    render(env)
    T += 1
  end
  @info("Episode $i finished after $T steps. Total reward: $R")
end
close(env)