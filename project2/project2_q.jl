using Graphs
using Printf
using CSV
using DataFrames
using Random
using SpecialFunctions
using LinearAlgebra
using IterTools

function write_small_policy(file_name::String)
    open(file_name, "w") do f
        for si in 1:100
            write(f, @sprintf("%d\n", small_policy(si)))
        end
    end
end

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

function (p::EpsilonGreedyExploration)(model, s) 
    A, epsilon = model.A, p.epsilon
    if rand() < epsilon
        return rand(A)
    end
    Q(s,a) = lookahead(model, s, a) 
    return argmax(a->Q(s,a), A)
end

struct MDP
    gamma # discount factor
    S # state space
    A # action space
    T # transition function 
    R # reward function
    TR # sample transition and reward
end 

function simulate(P::MDP, model, p, h, s) 
    for i in 1:h
        a = p(model, s)
        sp, r = P.TR(s, a)
        update!(model, s, a, r, sp)
        s = sp
    end
end

function make_small_policy(infile, outfile)
    data = CSV.read(infile, DataFrame)

    gamma = 0.95 # discount factor
    S = 1:100 # state space
    A = 1:4 # action space
    T = 0 # transition function 
    R = 0 # reward function
    P = MDP(gamma, S, A, T, R, TR)

    Q = zeros(length(P.S), length(P.A))
    alpha = 0.2 # learning rate
    model = QLearning(P.S, P.A, P.gamma, Q, alpha) 
    epsilon = 0.1 # probability of random action 
    p = EpsilonGreedyExploration(epsilon)
    k = 20 # number of steps to simulate 
    s = 1 # initial state
    simulate(P, model, p, k, s)
end

make_small_policy("./data/small.csv", "./output/small.policy")


# TR_mat = Array{Any}(undef, 100, 4)
    # for i in 1:length(data.s)
    #     s,a = data.s[i],data.a[i]
    #     if TR_mat[s,a] == 
    #         TR_mat[data.s[i],data.a[i]] = 
    #     end
    # end


    # data[!, :col] = 1:length(data.s)
    # TR_mat = Array{Any}(undef, 100, 4)
    # for i in 1:100
    #     datas = data[findall(in([s]), data.s), :]
    #     for a in 1:4
    #         datasa = datas[findall(in([a]), datas.a), :]
    #         TR_mat[s,a] = datasa.col
    # end

    # TR(s,a) = rand(TR_mat[s,a])
