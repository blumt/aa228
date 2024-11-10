using Graphs
using Printf
using CSV
using DataFrames
using Random
using SpecialFunctions
using LinearAlgebra
using IterTools

outfile = "./output/medium.policy"

open(outfile, "w") do f
    for s in 1:50000
        policy = rand([1,2,3,4,5,6,7])
        write(f, @sprintf("%d\n", policy))
    end
end