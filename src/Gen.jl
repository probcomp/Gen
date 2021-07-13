#__precompile__(false)

module Gen

const generated_functions = []

"""
    load_generated_functions(__module__=Main)

Permit use of generative functions written in the static modeling language up
to this point. Functions are loaded into Main by default.
"""
function load_generated_functions(__module__::Module=Main)
    for function_defn in generated_functions
        Core.eval(__module__, function_defn)
    end
end

"""
    @load_generated_functions

Permit use of generative functions written in the static modeling language up
to this point. Functions are loaded into the calling module.

This macro is intended to be called as a
[top-level expression](https://docs.julialang.org/en/v1/manual/modules/)
only; use in non–top-level scopes may result in incorrect behavior.
"""
macro load_generated_functions()
    for function_defn in generated_functions
        Core.eval(__module__, function_defn)
    end
end

export load_generated_functions, @load_generated_functions

# built-in extensions to the reverse mode AD
include("backprop.jl")

# addresses and address selections
include("address.jl")

# abstract and built-in concrete choice map data types
include("choice_map.jl")

# a homogeneous trie data type (not for use as choice map)
include("trie.jl")

# generative function interface
include("gen_fn_interface.jl")

# serialization/deserialization for traces
include("serialization.jl")

# built-in data types for arg-diff and ret-diff values
include("diff.jl")

# built-in probability disributions
include("modeling_library/modeling_library.jl")

# optimization of trainable parameters
include("optimization.jl")

# dynamic embedded generative function
include("dynamic/dynamic.jl")

# static IR generative function
include("static_ir/static_ir.jl")

# optimization for built-in generative functions (dynamic and static IR)
include("builtin_optimization.jl")

# DSLs for defining dynamic embedded and static IR generative functions
# 'Dynamic DSL' and 'Static DSL'
include("dsl/dsl.jl")

# inference and learning library
include("inference/inference.jl")

end # module Gen
