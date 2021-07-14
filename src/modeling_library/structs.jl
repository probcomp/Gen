export StructDiff, Construct, GetField

"Represents differences in the fields of user-defined structs."
struct StructDiff{T, D <: Tuple{Vararg{<:Diff}}} <: Diff
    diffs::D
    function StructDiff{T}(diffs::D) where {T, D}
        return new{T, D}(diffs)
    end
end

StructDiff{T}(diffs::Diff...) where {T} = StructDiff{T}(diffs)

function get_field_diff(diff::StructDiff, fieldname::Symbol)
    return _static_get_field_diff(diff, Val(fieldname))
end

@generated function _static_get_field_diff(diff::StructDiff{T}, ::Val{F}) where {T, F}
    idx = findfirst(fieldnames(T) .== F)
    return :(diff.diffs[$idx])
end

"""
    gen_fn = Construct(type::Type)

Constructs instances of user-defined composite types while supporting
incremental computation. Changes to fields are propogated accordingly.
"""
struct Construct{T} <: CustomUpdateGF{T, T} end

Construct(type::Type) = Construct{type}()

function apply_with_state(::Construct{T}, args) where {T}
    obj = T(args...)
    return (obj, obj)
end

function update_with_state(::Construct{T}, obj, args,
                           argdiffs::Tuple{Vararg{NoChange}}) where {T}
    return (obj, obj, NoChange())
end

function update_with_state(::Construct{T}, obj, args,
                           argdiffs::Tuple{Vararg{UnknownChange}}) where {T}
    new_obj = T(args...)
    return (new_obj, new_obj, UnknownChange())
end

@generated function update_with_state(::Construct{T}, obj, args,
                                      argdiffs::Tuple) where {T}
    atypes, ftypes = fieldtypes(args), fieldtypes(T)
    constructed_by_field = length(atypes) == length(ftypes) &&
        hasmethod(T, atypes) && all(map(zip(atypes, ftypes)) do (AT, FT)
            return AT <: FT || hasmethod(convert, (Type{FT}, AT))
        end)
    if constructed_by_field
        return quote
            new_obj = T(args...)
            return (new_obj, new_obj, StructDiff{T}(argdiffs))
        end
    else
        return quote
            new_obj = T(args...)
            return (new_obj, new_obj, UnknownChange())
        end
    end
end

"""
    gen_fn = GetField(type::Type, fieldname::Symbol)

Returns the field value of a user-defined composite type while supporting
incremental computation. Changes to fields are propagated accordingly.
"""
struct GetField{T, F, FT} <: CustomUpdateGF{FT, Nothing}
    function GetField{T, F}() where {T, F}
        return new{T, F, fieldtype(T, F)}()
    end
end

GetField(type::Type, fieldname::Symbol) = GetField{type, fieldname}()

function apply_with_state(::GetField{T, F}, (obj,)::Tuple{T}) where {T, F}
    return (getfield(obj, F), nothing)
end

function update_with_state(::GetField{T, F}, _, (obj,),
                           (diff,)::Tuple{NoChange}) where {T, F}
    return (nothing, getfield(obj, F), NoChange())
end

function update_with_state(::GetField{T, F}, _, (obj,),
                           (diff,)::Tuple{StructDiff{T}}) where {T, F}
    return (nothing, getfield(obj, F), get_field_diff(diff, F))
end

function update_with_state(::GetField{T, F}, _, (obj,),
                           (diff,)::Tuple{UnknownChange()}) where {T, F}
    return (nothing, getfield(obj, F), UnknownChange())
end
