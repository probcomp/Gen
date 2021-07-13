function _record_to_serializable(r::ChoiceOrCallRecord{T}) where {T <: Trace}
    @assert !r.is_choice
    return ChoiceOrCallRecord(to_serializable_trace(r.subtrace_or_retval), r.score, r.noise, r.is_choice)
end
function _record_to_serializable(r::ChoiceOrCallRecord)
    @assert r.is_choice
    return r
end
function _record_from_serializable(r::ChoiceOrCallRecord{T}, gf::GenerativeFunction) where {T <: SerializableTrace}
    @assert !r.is_choice
    return ChoiceOrCallRecord(from_serializable_trace(r.subtrace_or_retval, gf), r.score, r.noise, r.is_choice)
end
function _record_from_serializable(r::ChoiceOrCallRecord, dist::Distribution)
    @assert r.is_choice
    return r
end
function _trie_to_serializable(trie::Trie)
    triemap(trie, identity, _record_to_serializable)
end
function to_serializable_trace(tr::DynamicDSLTrace)
    return GenericSerializableTrace(
        _trie_to_serializable(tr.trie),
        (tr.isempty, tr.score, tr.noise, tr.args, tr.retval)
    )
end

# since a Dynamic Gen Function doesn't store
# what sub-generative-function is at which address,
# we have to run the generative function to get access to this!
mutable struct GFDeserializeState
    trace::DynamicDSLTrace
    serialized::GenericSerializableTrace
end
function from_serializable_trace(st::GenericSerializableTrace, gen_fn::DynamicDSLFunction{T}) where T
    trace = DynamicDSLTrace{T}(gen_fn, Trie{Any, ChoiceOrCallRecord}(), st.properties...)
    state = GFDeserializeState(trace, st)
    exec(gen_fn, state, trace.args)
    return trace
end
function traceat(state::GFDeserializeState, dist_or_gen_fn, args, key)
    record = _record_from_serializable(state.serialized.subtraces[key], dist_or_gen_fn)
    state.trace.trie[key] = record
    return record.is_choice ? record.subtrace_or_retval : get_retval(record.subtrace_or_retval)
end
function splice(state::GFDeserializeState, gf::DynamicDSLFunction, args::Tuple)
    return exec(gf, state, args)
end
read_param(::GFDeserializeState, ::Symbol) = nothing