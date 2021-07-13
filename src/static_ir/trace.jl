######################
# assignment wrapper #
######################

struct StaticIRTraceAssmt{T} <: ChoiceMap
    trace::T
end

function get_schema end

@inline get_address_schema(::Type{StaticIRTraceAssmt{T}}) where {T} = get_schema(T)

@inline Base.isempty(choices::StaticIRTraceAssmt) = isempty(choices.trace)

@inline static_has_value(choices::StaticIRTraceAssmt, key) = false

@inline function get_value(choices::StaticIRTraceAssmt, key::Symbol)
    static_get_value(choices, Val(key))
end

@inline function has_value(choices::StaticIRTraceAssmt, key::Symbol)
    static_has_value(choices, Val(key))
end

@inline function get_submap(choices::StaticIRTraceAssmt, key::Symbol)
    static_get_submap(choices, Val(key))
end
static_get_submap(::StaticIRTraceAssmt, ::Val) = EmptyChoiceMap()

@inline get_value(choices::StaticIRTraceAssmt, addr::Pair) = _get_value(choices, addr)
@inline has_value(choices::StaticIRTraceAssmt, addr::Pair) = _has_value(choices, addr)
@inline get_submap(choices::StaticIRTraceAssmt, addr::Pair) = _get_submap(choices, addr)

#########################
# trace type generation #
#########################

abstract type StaticIRTrace <: Trace end

@inline function static_get_subtrace(trace::StaticIRTrace, addr)
    error("Not implemented")
end

@inline static_haskey(trace::StaticIRTrace, ::Val) = false
 Base.haskey(trace::StaticIRTrace, key) = Gen.static_haskey(trace, Val(key))

@inline function Base.getindex(trace::StaticIRTrace, addr)
    Gen.static_getindex(trace, Val(addr))
end
@inline function Base.getindex(trace::StaticIRTrace, addr::Pair)
    first, rest = addr
    return Gen.static_get_subtrace(trace, Val(first))[rest]
end

const arg_prefix = gensym("arg")
const choice_value_prefix = gensym("choice_value")
const choice_score_prefix = gensym("choice_score")
const subtrace_prefix = gensym("subtrace")
const julia_prefix = gensym("julia_prefix")

function get_value_fieldname(node::ArgumentNode)
    Symbol("$(arg_prefix)_$(node.name)")
end

function get_value_fieldname(node::RandomChoiceNode)
    Symbol("$(choice_value_prefix)_$(node.addr)")
end

function get_value_fieldname(node::JuliaNode)
    Symbol("$(julia_prefix)_$(node.name)")
end

function get_score_fieldname(node::RandomChoiceNode)
    Symbol("$(choice_score_prefix)_$(node.addr)")
end

function get_subtrace_fieldname(node::GenerativeFunctionCallNode)
    Symbol("$(subtrace_prefix)_$(node.addr)")
end

const num_nonempty_fieldname = gensym("num_nonempty")
const total_score_fieldname = gensym("score")
const total_noise_fieldname = gensym("noise")
const return_value_fieldname = gensym("retval")

struct TraceField
    fieldname::Symbol
    typ::Union{Symbol,Expr,QuoteNode}
    holds_subtrace::Bool
end
TraceField(f, t) = TraceField(f, t, false)

function get_trace_fields(ir::StaticIR, options::StaticIRGenerativeFunctionOptions)
    fields = TraceField[]
    for node in ir.arg_nodes
        fieldname = get_value_fieldname(node)
        push!(fields, TraceField(fieldname, node.typ))
    end
    for node in ir.choice_nodes
        value_fieldname = get_value_fieldname(node)
        push!(fields, TraceField(value_fieldname, node.typ))
        score_fieldname = get_score_fieldname(node)
        push!(fields, TraceField(score_fieldname, QuoteNode(Float64)))
    end
    for node in ir.call_nodes
        subtrace_fieldname = get_subtrace_fieldname(node)
        subtrace_type = QuoteNode(get_trace_type(node.generative_function))
        push!(fields, TraceField(subtrace_fieldname, subtrace_type, true))
    end
    if options.cache_julia_nodes
        for node in ir.julia_nodes
            fieldname = get_value_fieldname(node)
            push!(fields, TraceField(fieldname, node.typ))
        end
    end
    push!(fields, TraceField(total_score_fieldname, QuoteNode(Float64)))
    push!(fields, TraceField(total_noise_fieldname, QuoteNode(Float64)))
    push!(fields, TraceField(num_nonempty_fieldname, QuoteNode(Int)))
    push!(fields, TraceField(return_value_fieldname, ir.return_node.typ))
    return fields
end

const static_ir_gen_fn_ref = gensym("gen_fn")

function generate_trace_struct(ir::StaticIR, trace_struct_name::Symbol, options::StaticIRGenerativeFunctionOptions)
    mutable = false
    fields = get_trace_fields(ir, options)
    field_exprs = map((f) -> Expr(:(::), f.fieldname, f.typ), fields)
    return (
        fields,
        Expr(:struct, mutable, Expr(:(<:), trace_struct_name, QuoteNode(StaticIRTrace)),
         Expr(:block, field_exprs..., Expr(:(::), static_ir_gen_fn_ref, QuoteNode(Any))))
   )
end

function generate_serialization_methods(ir::StaticIR, trace_struct_name::Symbol, gen_fn_typename::Symbol, fields)
    to_subtraces_exprs = [
        :($(GlobalRef(Gen, :to_serializable_trace))(tr.$(field.fieldname)))
        for field in fields if field.holds_subtrace
    ]
    to_properties_exprs = [:(tr.$(field.fieldname)) for field in fields if !field.holds_subtrace]
    
    # fields will have a bunch of properties, then the subtraces, then more properties
    num_initial_props = 0
    for field in fields
        if !field.holds_subtrace
            num_initial_props += 1
        else
            break;
        end
    end
    
    gen_fns = [node.generative_function for node in ir.call_nodes]
    
    quote
        function $(GlobalRef(Gen, :to_serializable_trace))(tr::$trace_struct_name)
            return $(GlobalRef(Gen, :GenericSerializableTrace))(
                $(Expr(:tuple, to_subtraces_exprs...)),
                $(Expr(:tuple, to_properties_exprs...))
            )
        end
        function $(GlobalRef(Gen, :from_serializable_trace))(
            st::$(GlobalRef(Gen, :GenericSerializableTrace)),
            gf::$gen_fn_typename
        )
            return $trace_struct_name(
                st.properties[1:$num_initial_props]...,
                (
                    $(GlobalRef(Gen, :from_serializable_trace))(args...)
                    for args in zip(st.subtraces, $gen_fns)
                )...,
                st.properties[$(num_initial_props + 1):end]...,
                gf
            )
        end
    end
end

function generate_isempty(trace_struct_name::Symbol)
    Expr(:function,
        Expr(:call, :(Base.isempty), :(trace::$trace_struct_name)),
        Expr(:block, :(trace.$num_nonempty_fieldname == 0)))
end

function generate_get_score(trace_struct_name::Symbol)
    Expr(:function,
        Expr(:call, GlobalRef(Gen, :get_score), :(trace::$trace_struct_name)),
        Expr(:block, :(trace.$total_score_fieldname)))
end

function generate_get_args(ir::StaticIR, trace_struct_name::Symbol)
    args = Expr(:tuple, [:(trace.$(get_value_fieldname(node)))
                         for node in ir.arg_nodes]...)
    Expr(:function,
        Expr(:call, GlobalRef(Gen, :get_args), :(trace::$trace_struct_name)),
        Expr(:block, args))
end

function generate_get_retval(ir::StaticIR, trace_struct_name::Symbol)
    Expr(:function,
        Expr(:call, GlobalRef(Gen, :get_retval), :(trace::$trace_struct_name)),
        Expr(:block, :(trace.$return_value_fieldname)))
end

function generate_get_choices(trace_struct_name::Symbol)
    Expr(:function,
        Expr(:call, GlobalRef(Gen, :get_choices), :(trace::$trace_struct_name)),
        Expr(:if, :(!isempty(trace)),
            :($(QuoteNode(StaticIRTraceAssmt))(trace)),
            :($(QuoteNode(EmptyChoiceMap))())))
end

function generate_get_values_shallow(ir::StaticIR, trace_struct_name::Symbol)
    elements = []
    for node in ir.choice_nodes
        addr = node.addr
        value = :(choices.trace.$(get_value_fieldname(node)))
        push!(elements, :(($(QuoteNode(addr)), $value)))
    end
    Expr(:function,
        Expr(:call, GlobalRef(Gen, :get_values_shallow),
                    :(choices::$(QuoteNode(StaticIRTraceAssmt)){$trace_struct_name})),
        Expr(:block, Expr(:tuple, elements...)))
end

function generate_get_submaps_shallow(ir::StaticIR, trace_struct_name::Symbol)
    elements = []
    for node in ir.call_nodes
        addr = node.addr
        subtrace = :(choices.trace.$(get_subtrace_fieldname(node)))
        push!(elements, :(($(QuoteNode(addr)), $(GlobalRef(Gen, :get_choices))($subtrace))))
    end
    Expr(:function,
        Expr(:call, GlobalRef(Gen, :get_submaps_shallow),
                    :(choices::$(QuoteNode(StaticIRTraceAssmt)){$trace_struct_name})),
        Expr(:block, Expr(:tuple, elements...)))
end

function generate_getindex(ir::StaticIR, trace_struct_name::Symbol)
    get_subtrace_exprs = Expr[]
    for node in ir.call_nodes
        push!(get_subtrace_exprs,
            quote
                function $(GlobalRef(Gen, :static_get_subtrace))(trace::$trace_struct_name, ::Val{$(QuoteNode(node.addr))})
                    return trace.$(get_subtrace_fieldname(node))
                end
            end
        )
    end

    call_getindex_exprs = Expr[]
    for node in ir.call_nodes
        push!(call_getindex_exprs,
            quote
                function $(GlobalRef(Gen, :static_getindex))(trace::$trace_struct_name, ::Val{$(QuoteNode(node.addr))})
                    return $(GlobalRef(Gen, :get_retval))(trace.$(get_subtrace_fieldname(node)))
                end
            end
        )
    end

    choice_getindex_exprs = Expr[]
    for node in ir.choice_nodes
        push!(choice_getindex_exprs,
            quote
                function $(GlobalRef(Gen, :static_getindex))(trace::$trace_struct_name, ::Val{$(QuoteNode(node.addr))})
                    return trace.$(get_value_fieldname(node))
                end
            end
        )
    end

    return [get_subtrace_exprs; call_getindex_exprs; choice_getindex_exprs]
end

function generate_static_get_value(ir::StaticIR, trace_struct_name::Symbol)
    methods = Expr[]
    for node in ir.choice_nodes
        push!(methods, Expr(:function,
            Expr(:call, GlobalRef(Gen, :static_get_value),
                        :(choices::$(QuoteNode(StaticIRTraceAssmt)){$trace_struct_name}),
                        :(::Val{$(QuoteNode(node.addr))})),
            Expr(:block, :(choices.trace.$(get_value_fieldname(node))))))
    end
    methods
end

function generate_static_has_value(ir::StaticIR, trace_struct_name::Symbol)
    methods = Expr[]
    for node in ir.choice_nodes
        push!(methods, Expr(:function,
            Expr(:call, GlobalRef(Gen, :static_has_value),
                        :(choices::$(QuoteNode(StaticIRTraceAssmt)){$trace_struct_name}),
                        :(::Val{$(QuoteNode(node.addr))})),
            Expr(:block, :(true))))
    end
    methods
end

function generate_static_get_submap(ir::StaticIR, trace_struct_name::Symbol)
    methods = Expr[]
    for node in ir.call_nodes
        push!(methods, Expr(:function,
            Expr(:call, GlobalRef(Gen, :static_get_submap),
                        :(choices::$(QuoteNode(StaticIRTraceAssmt)){$trace_struct_name}),
                        :(::Val{$(QuoteNode(node.addr))})),
            Expr(:block,
                :($(GlobalRef(Gen, :get_choices))(choices.trace.$(get_subtrace_fieldname(node)))))))
    end

    # throw a KeyError if get_submap is run on an address containing a value
    for node in ir.choice_nodes
         push!(methods, Expr(:function,
            Expr(:call, GlobalRef(Gen, :static_get_submap),
                        :(choices::$(QuoteNode(StaticIRTraceAssmt)){$trace_struct_name}),
                        :(::Val{$(QuoteNode(node.addr))})),
            Expr(:block, :(throw(KeyError($(QuoteNode(node.addr))))))))
    end
    methods
end

function generate_get_schema(ir::StaticIR, trace_struct_name::Symbol)
    choice_addrs = [QuoteNode(node.addr) for node in ir.choice_nodes]
    call_addrs = [QuoteNode(node.addr) for node in ir.call_nodes]
    addrs = vcat(choice_addrs, call_addrs)
    Expr(:function,
        Expr(:call, GlobalRef(Gen, :get_schema), :(::Type{$trace_struct_name})),
        Expr(:block,
            :($(QuoteNode(StaticAddressSchema))(
                Set{Symbol}([$(addrs...)])))))
end

function generate_trace_type_and_methods(ir::StaticIR, name::Symbol, options::StaticIRGenerativeFunctionOptions)
    trace_struct_name = gensym("StaticIRTrace_$name")
    (fields, trace_struct_expr) = generate_trace_struct(ir, trace_struct_name, options)
    isempty_expr = generate_isempty(trace_struct_name)
    get_score_expr = generate_get_score(trace_struct_name)
    get_args_expr = generate_get_args(ir, trace_struct_name)
    get_retval_expr = generate_get_retval(ir, trace_struct_name)
    get_choices_expr = generate_get_choices(trace_struct_name)
    get_schema_expr = generate_get_schema(ir, trace_struct_name)
    get_values_shallow_expr = generate_get_values_shallow(ir, trace_struct_name)
    get_submaps_shallow_expr = generate_get_submaps_shallow(ir, trace_struct_name)
    static_get_value_exprs = generate_static_get_value(ir, trace_struct_name)
    static_has_value_exprs = generate_static_has_value(ir, trace_struct_name)
    static_get_submap_exprs = generate_static_get_submap(ir, trace_struct_name)
    getindex_exprs = generate_getindex(ir, trace_struct_name)

    exprs = Expr(:block, trace_struct_expr, isempty_expr, get_score_expr,
                 get_args_expr, get_retval_expr,
                 get_choices_expr, get_schema_expr, get_values_shallow_expr,
                 get_submaps_shallow_expr, static_get_value_exprs...,
                 static_has_value_exprs..., static_get_submap_exprs...,
                 getindex_exprs...
                )
    (exprs, trace_struct_name, fields)
end

export StaticIRTrace
