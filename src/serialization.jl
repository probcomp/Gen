using Serialization: serialize, deserialize

"""
    SerializableTrace

A representation of a `Trace` which can be serialized. Obtainable via `to_serializable_trace`.
This does not need to contain the `GenerativeFunction` which produced the trace;
to deserialize (using `from_serializable_trace`), the `GenerativeFunction` must be provided.
"""
abstract type SerializableTrace end

"""
    to_serializable_trace(trace::Trace)

Get a SerializableTrace representing the `trace` in a serializable manner.
"""
function to_serializable_trace(trace::Trace)
    error("Not implemented")
end

"""
    from_serializable_trace(st::SerializableTrace, fn::GenerativeFunction)

Get the trace of the given generative function encoded by the serializable trace object.
"""
function from_serializable_trace(::SerializableTrace, ::GenerativeFunction)
    error("Not implemented.")
end

"""
    serialize_trace(stream::IO, trace::Trace)
    serialize_trace(filename::AbstractString, trace::Trace)

Serialize the given trace to the given stream or file, by converting to a `SerializableTrace`.
"""
function serialize_trace(filename_or_io::Union{IO, AbstractString}, trace::Trace)
    return serialize(filename_or_io, to_serializable_trace(trace))
end

"""
    deserialize_trace(stream::IO, gen_fn::GenerativeFunction)
    deserialize_trace(filename::AbstractString, gen_fn::GenerativeFunction)

Deserialize the trace for the given generative function stored in the given stream or file
(as saved via `serialize_trace`).
"""
function deserialize_trace(filename_or_io::Union{IO, AbstractString}, gf::GenerativeFunction)
    return from_serializable_trace(deserialize(filename_or_io), gf)
end

"""
    GenericSerializableTrace <: SerializableTrace

A SerializableTrace which contains some subtraces which have been recursively converted
to `SerializableTrace`s, and some properties which are directly serializable.
"""
struct GenericSerializableTrace{S, P} <: SerializableTrace
    subtraces::S
    properties::P
end

export to_serializable_trace, from_serializable_trace, serialize_trace, deserialize_trace