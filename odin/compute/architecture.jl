#=
architecture.jl
- Julia version 1.0
- Author anderscs
- Date 2018-11-26
=#
using LinearAlgebra, Distributed, Printf, NPZ

function group_argmax(array::Array, n::Int)
    queue = zeros(n)
    max_group = zeros(n)
    for i in 1:length(array)
        a = array[i]
        qmin = minimum(queue)
        if a > qmin
            amin = argmin(queue)
            queue[amin] = a
            max_group[amin] = i
        end
    end
    return max_group
end



function greedy(constraint::Function, indexes::Array{Int}, m_l::Int, parallel::Bool=false, avalible_cores::Int=4)
    """
    Greedy selection of nodes
    """

    selected = Array{Int}([])
    plot = false
    choices = copy(indexes)

    for i in 1:m_l
        @printf("i = %d\n", i)

        calc(node::Int) = constraint(union(selected, node))

        @time values = pmap(calc, choices)


        greedy_choice = choices[argmax(values)]

        if plot
            values = sort(values)
            oplt.plot(values)
            oplt.show()
            # current_best = np.max(values)
        end

        selected = union(selected, [greedy_choice])
        choices = setdiff(choices, [greedy_choice])
        #@printf("selected = %s; choice = %s;\n", selected, greedy_choice)
    end

    return selected
end


function compute_index_set(layer, m_l, weights)
    data = load_data(model_name = "mnist_vgg2",
    prefix = "",
    group_name = "inter_layer_covariance")

    cov_list = data["cov"]
    cov = cov_list[layer + 1, :, :]
    # println("cov for $layer", cov[1:10,1:10])

    shape = size(cov)
    indexes = Array{Int}(1:shape[1])

    theta = 0.5
    W = weights
    #R_z = W' * inv(W * W') * W


    function C(j::Array{Int})
        f = setdiff(indexes, j)
        n = length(f)
        sig_inv = inv(cov[j, j])

        Id = Matrix{Float64}(I, n, n)
        R_z = Matrix{Float64}(I, n, n) # Projection matrix

        ch = theta * Id + (1 - theta) * R_z

        difference = tr(ch * cov[f, j] * sig_inv * cov[j, f])
        normalizer = tr(ch * cov[f, f])
        return difference / normalizer
    end

    j = greedy(C, indexes, m_l)

    return j
end

const results_dir = "results"

function load_data(group_name::String, model_name::String = "mnist_vgg2", prefix::String = "")

    data = Dict()
    dir = joinpath(results_dir, model_name, prefix, group_name)
    for (root, dirs, files) in walkdir(dir)
        for file in files
            parts = split(file, ".")
            println("loading $file in $root")
            if parts[2] == "npy"
                data[parts[1]] = npzread(joinpath(root, file))
            end
        end
    end

    return data
end

function save_data(data::Dict, group_name::String, model_name = "mnist_vgg2", prefix = "")
    dir = joinpath(results_dir, model_name, prefix, group_name)
    for (name, value) in data
        npzwrite(joinpath(root, name), value)
    end

    return data
end


function test()
    cov = Matrix{Float64}(I, 500, 500)
    m_l = 200
    weights = Matrix{Float64}(I, 500, 500)
    compute_index_set(cov, m_l, weights)
end

