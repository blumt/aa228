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

function beta(s) #simulate distance from flag for reward shaping function
    flag = 465
    return 100000.0 / abs(flag - mod(s-1,500))
end

function F(s,a,sp,gamma)
    return gamma * beta(sp) - beta(s)
end

function update!(model::QLearning, s, a, r, sp) 
    gamma, Q, alpha = model.gamma, model.Q, model.alpha
    f = F(s,a,sp,gamma)
    Q[s,a] += alpha*(r + f + gamma*maximum(Q[sp,:]) - Q[s,a])
    return model
end

gamma = 1 # discount factor
nstates = 50000
nactions = 7
S = 1:nstates # state space
A = 1:nactions # action space

# initialize Q
Q = fill(0.0, nstates, nactions)
alpha = .1
model = QLearning(S, A, gamma, Q, alpha)

# simulate based on data
infile = "./data/medium.csv"
outfile = "./output/medium.policy"
data = CSV.read(infile, DataFrame)
for i in 1:100
    for i in 1:size(data,1)
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
    end
end
