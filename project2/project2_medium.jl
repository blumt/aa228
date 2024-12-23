using Graphs
using Printf
using CSV
using DataFrames
using Random
using SpecialFunctions
using LinearAlgebra
using IterTools

mutable struct QLearning
    S # state space (assumes 1:nstates) 
    A # action space (assumes 1:nactions) 
    gamma # discount
    Q # action value function
    alpha # learning rate 
end 

lookahead(model::QLearning, s, a) = model.Q[s,a]

function F(s,sp,gamma)
    dist_sp = abs(465 - (sp %500))
    dist_s = abs(465 - (s %500))
    return gamma * (50000.0 / dist_sp) - (50000.0 / dist_s)
end

function update!(model::QLearning, s, a, r, sp) 
    gamma, Q, alpha = model.gamma, model.Q, model.alpha
    Q[s,a] += alpha * (r + F(s,sp,gamma) + gamma * maximum(Q[sp,:]) - Q[s,a])
    return model
end

gamma = .99 # discount factor
nstates = 50000
nactions = 7
S = 1:nstates # state space
A = 1:nactions # action space

# initialize Q
Q = fill(0.0, nstates, nactions)
alpha = 0.1
model = QLearning(S, A, gamma, Q, alpha)

# simulate based on data
infile = "./data/medium.csv"
outfile = "./output/medium.policy"
data = CSV.read(infile, DataFrame)
for j in 1:1000
    for i in 1:length(data.s)
        s = data.s[i]
        a = data.a[i]
        sp = data.sp[i]
        r = data.r[i]
        update!(model, s, a, r, sp)
    end
end

# make policy file by taking best Q value for each state, action
open(outfile, "w") do f
    for s in model.S
        policy = argmax(model.Q[s, :])
        write(f, @sprintf("%d\n", policy))
        # write(f, @sprintf("%d %d %d %d %d %d\n", model.Q[s,1], model.Q[s,2],model.Q[s,3], model.Q[s,4],model.Q[s,5], model.Q[s,6]))
    end
end
