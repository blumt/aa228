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

function update!(model::QLearning, s, a, r, sp) 
    gamma, Q, alpha = model.gamma, model.Q, model.alpha
    Q[s,a] += alpha*(r + gamma*maximum(Q[sp,:]) - Q[s,a])
    return model
end

gamma = 0.95 # discount factor
nstates = 100
nactions = 4
S = 1:nstates # state space
A = 1:nactions # action space

# initialize Q
Q = zeros(nstates, nactions)
alpha = .2
model = QLearning(S, A, gamma, Q, alpha)

# simulate based on data
infile = "./data/small.csv"
outfile = "./output/small.policy"
data = CSV.read(infile, DataFrame)
for i in 1:size(data,1)
    s = data.s[i]
    a = data.a[i]
    sp = data.sp[i]
    r = data.r[i]
    update!(model, s, a, r, sp)
end

# make policy file by taking best Q value for each state, action
open(outfile, "w") do f
    for s in model.S
        policy = argmax(model.Q[s, :])
        write(f, @sprintf("%d\n", policy))
    end
end
