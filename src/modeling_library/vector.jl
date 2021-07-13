using FunctionalCollections: PersistentVector, assoc, push, pop

###########################################
# trace used by vector-shaped combinators #
###########################################

"""

U is the type of the subtrace, R is the return value type for the kernel
"""
struct VectorTrace{GenFnType,T,U} <: Trace
    gen_fn::GenerativeFunction
    subtraces::PersistentVector{U}
    retval::PersistentVector{T}
    args::Tuple

    # number of active subtraces (may be less than length of subtraces)
    len::Int

    # number of active subtraces that are nonempty (used for has_choices)
    num_nonempty::Int

    score::Float64
    noise::Float64
end

function VectorTrace{GenFnType,T,U}(gen_fn::GenerativeFunction,
                                    subtraces::PersistentVector{U},
                                    retval::PersistentVector{T},
                                    args::Tuple, score::Float64, noise::Float64,
                                    len::Int, num_nonempty::Int) where {GenFnType,T,U}
    @assert length(subtraces) == length(retval)
    @assert length(subtraces) == len
    @assert num_nonempty >= 0
    VectorTrace{GenFnType,T,U}(gen_fn, subtraces, retval, args, len,
        num_nonempty, score, noise)
end

function VectorTrace{GenFnType,T,U}(gen_fn::GenerativeFunction, args::Tuple) where {GenFnType,T,U}
    subtraces = PersistentVector{U}()
    retvals = PersistentVector{T}()
    VectorTrace{GenFnType,T,U}(gen_fn, subtraces, retvals, args, 0, 0, 0., 0.)
end

# trace API

@inline get_choices(trace::VectorTrace) = VectorTraceChoiceMap(trace)
@inline get_retval(trace::VectorTrace) = trace.retval
@inline get_args(trace::VectorTrace) = trace.args
@inline get_score(trace::VectorTrace) = trace.score
@inline get_gen_fn(trace::VectorTrace) = trace.gen_fn

@inline function Base.getindex(trace::VectorTrace{GenFnType, T, U}, addr::Pair) where {GenFnType, T, U}
    (first, rest) = addr
    subtrace = trace.subtraces[first]
    subtrace[rest]
end

@inline function Base.getindex(trace::VectorTrace, addr)
    # we expose the return values in the auxiliary state
    trace.retval[addr]
end


function project(trace::VectorTrace, selection::Selection)
    weight = 0.
    for key=1:trace.len
        subselection = selection[key]
        weight += project(trace.subtraces[key], subselection)
    end
    weight
end
project(trace::VectorTrace, ::EmptySelection) = trace.noise

struct VectorTraceChoiceMap{GenFnType, T, U} <: ChoiceMap
    trace::VectorTrace{GenFnType, T, U}
end

@inline Base.isempty(assignment::VectorTraceChoiceMap) = assignment.trace.num_nonempty == 0
@inline get_address_schema(::Type{VectorTraceChoiceMap}) = VectorAddressSchema()

@inline function get_submap(choices::VectorTraceChoiceMap, addr::Int)
    if addr <= choices.trace.len
        get_choices(choices.trace.subtraces[addr])
    else
        EmptyChoiceMap()
    end
end

@inline function get_submaps_shallow(choices::VectorTraceChoiceMap)
    ((i, get_choices(choices.trace.subtraces[i])) for i=1:choices.trace.len)
end

@inline get_submap(choices::VectorTraceChoiceMap, addr::Pair) = _get_submap(choices, addr)
@inline get_value(choices::VectorTraceChoiceMap, addr::Pair) = _get_value(choices, addr)
@inline has_value(choices::VectorTraceChoiceMap, addr::Pair) = _has_value(choices, addr)
@inline get_values_shallow(::VectorTraceChoiceMap) = ()


############################################
# code shared by vector-shaped combinators #
############################################

function get_retained_and_constrained(constraints::ChoiceMap, prev_length::Int, new_length::Int)
    keys = Set{Int}()
    for (key::Int, _) in get_submaps_shallow(constraints)
        if key > 0 && key <= new_length
            push!(keys, key)
        else
            error("Constrained address does not exist: $key")
        end
    end
    keys
end

function get_retained_and_selected(selection::EmptySelection, prev_length::Int, new_length::Int)
    Set{Int}()
end

function get_retained_and_selected(selection::HierarchicalSelection, prev_length::Int, new_length::Int)
    keys = Set{Int}()
    for (key::Int, _) in get_subselections(selection)
        if key > 0 && key <= new_length
            push!(keys, key)
        else
            error("Selected address does not exist: $key")
        end
    end
    keys
end

function get_retained_and_selected(selection::AllSelection, prev_length::Int, new_length::Int)
    Set{Int}(1:min(prev_length, new_length))
end

function vector_compute_retdiff(updated_retdiffs::Dict{Int,Diff}, new_length::Int, prev_length::Int)
    if new_length == prev_length && length(updated_retdiffs) == 0
        NoChange()
    else
        VectorDiff(new_length, prev_length, updated_retdiffs)
    end
end

function vector_update_delete(new_length::Int, prev_length::Int,
                                 prev_trace::VectorTrace)
    num_nonempty = prev_trace.num_nonempty
    discard = choicemap()
    score_decrement = 0.
    noise_decrement = 0.
    for key=new_length+1:prev_length
        subtrace = prev_trace.subtraces[key]
        score_decrement += get_score(subtrace)
        noise_decrement += project(subtrace, EmptySelection())
        if !isempty(get_choices(subtrace))
            num_nonempty -= 1
        end
        @assert num_nonempty >= 0
        set_submap!(discard, key, get_choices(subtrace))
    end
    return (discard, num_nonempty, score_decrement, noise_decrement)
end

function vector_regenerate_delete(new_length::Int, prev_length::Int,
                                  prev_trace::VectorTrace)
    num_nonempty = prev_trace.num_nonempty
    score_decrement = 0.
    noise_decrement = 0.
    for key=new_length+1:prev_length
        subtrace = prev_trace.subtraces[key]
        score_decrement += get_score(subtrace)
        noise_decrement += project(subtrace, EmptySelection())
        if !isempty(get_choices(subtrace))
            num_nonempty -= 1
        end
        @assert num_nonempty >= 0
    end
    return (num_nonempty, score_decrement, noise_decrement)
end

function vector_remove_deleted_applications(subtraces, retval, prev_length, new_length)
    for i=new_length+1:prev_length
        subtraces = pop(subtraces)
        retval = pop(retval)
    end
    (subtraces, retval)
end

#################
# Serialization #
#################
function to_serializable_trace(trace::VectorTrace)
    GenericSerializableTrace(
        [to_serializable_trace(st) for st in trace.subtraces],
        (trace.retval, trace.args, trace.len, trace.num_nonempty, trace.score, trace.noise)
    )
end
function from_serializable_trace(st::GenericSerializableTrace, gf::GenerativeFunction{<:Any, VectorTrace{GenFnType, T, U}}) where {GenFnType, T, U}
    subtraces = PersistentVector{U}(
        [from_serializable_trace(serialized_subtrace, _gen_fn_at_addr(gf, i))
        for (i, serialized_subtrace) in enumerate(st.subtraces)]
    )
    get_trace_type(gf)(gf, subtraces, st.properties...)
end