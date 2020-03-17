const STATIC_DSL_GRAD = Symbol("@grad")
const STATIC_DSL_TRACE = Symbol("@trace")
const STATIC_DSL_PARAM = Symbol("@param")

function static_dsl_syntax_error(expr, msg="")
    error("Syntax error when parsing static DSL function at $expr. $msg")
end

function parse_typed_var(expr)
    if MacroTools.@capture(expr, var_Symbol)
        return (var, QuoteNode(Any))
    elseif MacroTools.@capture(expr, var_Symbol::typ_)
        return (var, typ)
    else
        static_dsl_syntax_error(expr)
    end
end

function resolve_symbols(bindings::Dict{Symbol,Symbol}, symbol::Symbol)
    resolved = Dict{Symbol,Symbol}()
    if haskey(bindings, symbol)
        resolved[symbol] = bindings[symbol]
    end
    resolved
end

function resolve_symbols(bindings::Dict{Symbol,Symbol}, expr::Expr)
    resolved = Dict{Symbol,Symbol}()
    if expr.head == :(.)
        merge!(resolved, resolve_symbols(bindings, expr.args[1]))
    else
        for arg in expr.args
            merge!(resolved, resolve_symbols(bindings, arg))
        end
    end
    resolved
end

function resolve_symbols(bindings::Dict{Symbol,Symbol}, value)
    Dict{Symbol,Symbol}()
end

# the IR builder needs to contain a bindings map from symbol to IRNode, to
# provide us with input_nodes.

# the macro expansion also needs a bindings set of symbols to resolve from, so that
# we can then insert the loo

function parse_julia_expr!(stmts, bindings, name::Symbol, expr::Expr,
                           typ::Union{Symbol,Expr,QuoteNode})
    resolved = resolve_symbols(bindings, expr)
    inputs = collect(resolved)
    input_vars = map((x) -> esc(x[1]), inputs)
    input_nodes = map((x) -> esc(x[2]), inputs)
    fn = Expr(:function, Expr(:tuple, input_vars...), esc(expr))
    node = gensym(name)
    push!(stmts, :($(esc(node)) = add_julia_node!(
        builder, $fn, inputs=[$(input_nodes...)], name=$(QuoteNode(name)),
        typ=$(QuoteNode(typ)))))
    return node
end

function parse_julia_expr!(stmts, bindings, name::Symbol, var::Symbol,
                           typ::Union{Symbol,Expr,QuoteNode})
    if haskey(bindings, var)
        # don't create a new Julia node, just use the existing node
        return bindings[var]
    end
    return parse_julia_expr!(stmts, bindings, name, Expr(:block, var), typ)
end

function parse_julia_expr!(stmts, bindings, name::Symbol, var::QuoteNode,
                           typ::Union{Symbol,Expr,QuoteNode})
    fn = Expr(:function, Expr(:tuple), var)
    node = gensym(name)
    push!(stmts, :($(esc(node)) = add_julia_node!(
        builder, $fn, inputs=[], name=$(QuoteNode(name)),
        typ=$(QuoteNode(typ)))))
    return node
end

function parse_julia_expr!(stmts, bindings, name::Symbol, value,
                           typ::Union{Symbol,Expr,QuoteNode})
    fn = Expr(:function, Expr(:tuple), QuoteNode(value))
    node = gensym(name)
    push!(stmts, :($(esc(node)) = add_julia_node!(
        builder, $fn, inputs=[], name=$(QuoteNode(name)),
        typ=$(QuoteNode(typ)))))
    return node
end

function parse_assignment!(stmts, bindings, lhs, rhs)
    if isa(lhs, Expr) && lhs.head == :tuple
        # Recursively handle tuple assignments
        name, typ = gen_node_name(rhs), QuoteNode(Any)
        node = parse_julia_expr!(stmts, bindings, name, rhs, typ)
        bindings[name] = node
        for (i, lhs_i) in enumerate(lhs.args)
            # Assign lhs[i] = rhs[i]
            rhs_i = :($name[$i])
            parse_assignment!(stmts, bindings, lhs_i, rhs_i)
        end
    else
        # Handle single variable assignment (base case)
        (name::Symbol, typ) = parse_typed_var(lhs)
        # Create new name if variable is already bound
        if haskey(bindings, name) name = gensym(name) end
        node = parse_julia_expr!(stmts, bindings, name, rhs, typ)
        bindings[name] = node
    end
    # Return name of node to be processed by parent expressions
    return name
end

split_addr!(keys, addr_expr::QuoteNode) = push!(keys, addr_expr)
split_addr!(keys, addr_expr::Symbol) = push!(keys, addr_expr)

function split_addr!(keys, addr_expr::Expr)
    @assert MacroTools.@capture(addr_expr, fst_ => snd_)
    push!(keys, fst)
    split_addr!(keys, snd)
end

choice_or_call_at(gen_fn::GenerativeFunction, addr_typ) = call_at(gen_fn, addr_typ)
choice_or_call_at(dist::Distribution, addr_typ) = choice_at(dist, addr_typ)

gen_node_name(arg::Symbol) = gensym(arg)
gen_node_name(arg::QuoteNode) = gensym(repr(arg.value))
gen_node_name(arg::Expr) = gensym(arg.head)
gen_node_name(arg::Any) = gensym(repr(arg))

function parse_trace_expr!(stmts, bindings, fn, args, addr)
    expr_s = "$STATIC_DSL_TRACE($fn($(join(args, ", "))), $addr)"
    name = gen_node_name(addr) # Each @trace node is named after its address
    node = gen_node_name(addr) # Generate a variable name for the StaticIRNode
    bindings[name] = node
    if !isa(fn, Symbol)
        static_dsl_syntax_error(expr_s, "$fn is not a Symbol")
    end
    gen_fn_or_dist = gensym(fn)
    push!(stmts, :($(esc(gen_fn_or_dist)) = $(esc(fn))))

    keys = []
    split_addr!(keys, addr) # Split nested addresses
    if !(isa(keys[1], QuoteNode) && isa(keys[1].value, Symbol))
        static_dsl_syntax_error(addr, "$(keys[1].value) is not a Symbol")
    end
    addr = keys[1].value # Get top level address
    if length(keys) > 1
        for key in keys[2:end]
            # For each nesting level, wrap fn within choice_at / call_at
            push!(stmts, :($(esc(gen_fn_or_dist)) =
                choice_or_call_at($(esc(gen_fn_or_dist)), Any)))
        end
        # Append the nested addresses as arguments to choice_at / call_at
        args = [args; reverse(keys[2:end])]
    end

    inputs = []
    for arg_expr in args
        if MacroTools.@capture(arg_expr, x_...)
            static_dsl_syntax_error(expr_s, "Cannot splat in @trace call.")
        end
        # Create Julia node for each argument to gen_fn_or_dist
        arg_name = gen_node_name(arg_expr)
        push!(inputs, parse_julia_expr!(stmts, bindings,
                                        arg_name, arg_expr, QuoteNode(Any)))
    end

    # Add addr node
    push!(stmts, :($(esc(node)) = add_addr_node!(
        builder, $(esc(gen_fn_or_dist)), inputs=[$(map(esc, inputs)...)],
        addr=$(QuoteNode(addr)), name=$(QuoteNode(name)))))
    # Return the name of the newly created node
    return name
end

function parse_trainable_param!(stmts::Vector{Expr}, bindings, expr::Expr)
    (name::Symbol, typ) = parse_typed_var(expr)
    if haskey(bindings, name)
        static_dsl_syntax_error(expr, "Symbol $name already bound")
    end
    node = gensym(name)
    bindings[name] = node
    push!(stmts, :($(esc(node)) = add_trainable_param_node!(
        builder, $(QuoteNode(name)), typ=$(QuoteNode(typ)))))
    true
end

function parse_return!(stmts::Vector{Expr}, bindings, expr)
    if isa(expr, Symbol)
        if !haskey(bindings, expr)
            error("Tried to return $expr, which is not a locally bound variable")
        end
        node = bindings[expr]
    else
        name, typ = gensym("return"), QuoteNode(Any)
        node = parse_julia_expr!(stmts, bindings, name, expr, typ)
        bindings[name] = node
    end
    push!(stmts, :(set_return_node!(builder, $(esc(node)))))
    return Expr(:return, expr)
end

function parse_expr!(stmts, bindings, expr)
    if MacroTools.@capture(expr, @m_(f_(xs__), addr_)) && m == STATIC_DSL_TRACE
        # Parse "@trace(f(xs...), addr)" and return fresh variable
        parse_trace_expr!(stmts, bindings, f, xs, addr)
    elseif MacroTools.@capture(expr, @m_(f_(xs__))) && m == STATIC_DSL_TRACE
        # Throw error for @trace expression without address
        static_dsl_syntax_error(expr, "Address required.")
    elseif MacroTools.@capture(expr, @m_ e_) && m == STATIC_DSL_PARAM
        # Parse "@param var::T" and return var
        parse_trainable_param!(stmts, bindings, e)
    elseif MacroTools.@capture(expr, lhs_ = rhs_)
        # Parse "lhs = rhs" and return lhs
        parse_assignment!(stmts, bindings, lhs, rhs)
    elseif MacroTools.@capture(expr, return e_)
        # Parse "return expr" and return expr
        parse_return!(stmts, bindings, e)
    else
        expr
    end
end

function parse_static_dsl_function_body!(
    stmts::Vector{Expr}, bindings::Dict{Symbol,Symbol}, expr)
    # TODO use line number nodes to provide better error messages in generated code
    if !(isa(expr, Expr) && expr.head == :block)
        static_dsl_syntax_error(expr)
    end
    for line in expr.args
        MacroTools.postwalk(e -> parse_expr!(stmts, bindings, e), line)
    end
end

function make_static_gen_function(name, args, body, return_type, annotations)
    # generate code that builds the IR, then generates code from it and evaluates it
    stmts = Expr[]
    push!(stmts, :(bindings = Dict{Symbol, StaticIRNode}()))
    push!(stmts, :(builder = StaticIRBuilder())) # NOTE: we are relying on the gensym
    accepts_output_grad = DSL_RET_GRAD_ANNOTATION in annotations
    push!(stmts, :(set_accepts_output_grad!(builder, $(QuoteNode(accepts_output_grad)))))
    bindings = Dict{Symbol,Symbol}() # map from variable name to node name
    for arg in args
        if arg.default != nothing
            error("Default argument values not supported in the static DSL.")
        end
        node = gensym(arg.name)
        push!(stmts, :($(esc(node)) = add_argument_node!(
            builder, name=$(QuoteNode(arg.name)), typ=$(QuoteNode(arg.typ)),
            compute_grad=$(QuoteNode(DSL_ARG_GRAD_ANNOTATION in arg.annotations)))))
        bindings[arg.name] = node
    end
    parse_static_dsl_function_body!(stmts, bindings, body)
    push!(stmts, :(ir = build_ir(builder)))
    expr = gensym("gen_fn_defn")
    # note: use the eval() for the user's module, not Gen
    track_diffs = DSL_TRACK_DIFFS_ANNOTATION in annotations
    cache_julia_nodes = !(DSL_NO_JULIA_CACHE_ANNOTATION in annotations) # cache julia nodes by default
    options = StaticIRGenerativeFunctionOptions(track_diffs, cache_julia_nodes)
    push!(stmts, :(Core.@__doc__ $(esc(name)) = $(esc(:eval))(
        generate_generative_function(ir, $(QuoteNode(name)), $(QuoteNode(options))))))
    Expr(:block, stmts...)
end
