import Random

function logsumexp(arr::AbstractArray{T}) where {T <: Real}
    max_arr = maximum(arr)
    max_arr == -Inf ? -Inf : max_arr + log(sum(exp.(arr .- max_arr)))
end

function logsumexp(x1::Real, x2::Real)
    m = max(x1, x2)
    m == -Inf ? m : m + log(exp(x1 - m) + exp(x2 - m))
end

export logsumexp

include("trace_translators.jl")

# mcmc
include("kernel_dsl.jl")
include("mh.jl")
include("hmc.jl")
include("mala.jl")
include("elliptical_slice.jl")

include("importance.jl")
include("particle_filter.jl")
include("map_optimize.jl")
include("train.jl")
include("variational.jl")
include("ais.jl")
