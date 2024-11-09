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

mutable struct MaximumLikelihoodMDP
    S # state space (assumes 1:nstates) 
    A # action space (assumes 1:nactions) 
    N # transition count N(s,a,s′)
    rho # reward sum rho(s, a)
    gamma # discount
    U # value function
    planner
end

function lookahead(model::MaximumLikelihoodMDP, s, a) 
    S, U, gamma = model.S, model.U, model.gamma
    n = sum(model.N[s,a,:])
    if n == 0
        return 0.0
    end
    r = model.rho[s, a] / n
    T(s,a,s_prime) = model.N[s,a,s_prime] / n
    return r + gamma * sum(T(s,a,s_prime)*U[s_prime] for s_prime in S)
end

function backup(model::MaximumLikelihoodMDP, U, s)
    return maximum(lookahead(model, s, a) for a in model.A)
end
    
function update!(model::MaximumLikelihoodMDP, s, a, r, s_prime) 
    model.N[s,a,s_prime] += 1
    model.rho[s,a] += r
    update!(model.planner, model, s, a, r, s_prime)
    return model
end

struct MDP
    T # transition function (indexed by [S,A,S'])
    R # reward function (indexed by [S,A])
    gamma # discount factor
    # S # state space
    # A # action space
    # TR # sample transition and reward
end

function MDP(model::MaximumLikelihoodMDP)
    N, rho, S, A, gamma = model.N, model.rho, model.S, model.A, model.gamma
    T, R = similar(N), similar(rho)
    for s in S
        for a in A
            n = sum(N[s,a,:]) 
            if n == 0
                T[s,a,:] .= 0.0 
                R[s,a] = 0.0
            else
                T[s,a,:] = N[s,a,:] / n
                R[s,a] = rho[s,a] / n
            end
        end
    end
    return MDP(T, R, gamma)
end



struct RandomizedUpdate 
    m # number of updates
end

function update!(planner::RandomizedUpdate, model, s, a, r, s_prime) 
    U = model.U
    U[s] = backup(model, U, s) 
    for i in 1:planner.m
        s = rand(model.S)
        U[s] = backup(model, U, s)
    end
    return planner
end

# mutable struct EpsilonGreedyExploration 
#     epsilon # probability of random arm
# end
    
# function (pi::EpsilonGreedyExploration)(model::BanditModel) 
#     if rand() < pi.epsilon
#         return rand(eachindex(model.B))
#     else 
#         return argmax(mean.(model.B))
#     end
# end

function make_small_policy()
    infile = "./data/small.csv"
    data = CSV.read(infile, DataFrame)

    S_init = 1:100
    A_init = 1:4
    N_init = [0 for (s,a,sp) in product(S_init,A_init,S_init)]
    rho_init = [0 for (s,a) in product(S_init,A_init)]
    gamma = 0.95
    U = 0
    planner = RandomizedUpdate(10)
    small_model = MaximumLikelihoodMDP(S_init,A_init,N_init,rho_init,gamma,U,planner)

    n_iterations = length(data.s)
    for i in 1:n_iterations
        s, a, r, s_prime = data.s[i], data.a[i], data.r[i], data.sp[i]
        update!(small_model, s, a, r, s_prime)
    end

    small_mdp = MDP(small_model)
    
    small_policy(s) = pick_best
end
    # for y in 10:-1:1
    #     for x in 1:10
    #         @printf("%2d ", LinearIndices((10,10))[x,y])
    #     end
    #     println()
    # end

    # if s == 47
    #     return rand([1,2,3,4])
    # elseif s == 92
    #     return 4
    # elseif s == 83
    #     return 1
    # elseif s == 81
    #     return 2
    # elseif s == 82 
    #     return 4
    # end
    # if mod(s,10) == 7 # to the right of 47
    #     if s > 47
    #         return 4
    #     elseif s < 47
    #         return 3
    #     end
    # elseif mod(s,10) > 7 || mod(s,10) == 0 #to the left of 47
    #     return 1
    # else #
    #     return 2
    # end

write_small_policy("./output/small.policy")