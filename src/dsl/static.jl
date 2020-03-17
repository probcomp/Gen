const STATIC_DSL_GRAD = Symbol("@grad")
const STATIC_DSL_TRACE = Symbol("@trace")
const STATIC_DSL_PARAM = Symbol("@param")

function static_dsl_syntax_error(expr, msg="")
    error("Syntax error when parsing static DSL function at $expr. $msg")
end

function parse_lhs(lhs)
    if isa(lhs, Symbol)
        return (lhs, QuoteNode(Any))
    elseif isa(lhs, Expr) && lhs.head == :(::)
        return (lhs.args[1], lhs.args[2])
    else
        static_dsl_syntax_error(lhs)
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
    node = gensym(expr.head)
    push!(stmts, :($(esc(node)) = add_julia_node!(
        builder, $fn, inputs=[$(input_nodes...)], name=$(QuoteNode(name)),
        typ=$(QuoteNode(typ)))))
    node
end

function parse_julia_expr!(stmts, bindings, name::Symbol, var::Symbol,
                           typ::Union{Symbol,Expr,QuoteNode})
    if haskey(bindings, var)
        # don't create a new Julia node, just use the existing node
        return bindings[var]
    end
    parse_julia_expr!(stmts, bindings, name, Expr(:block, var), typ)
end

function parse_julia_expr!(stmts, bindings, name::Symbol, var::QuoteNode,
                           typ::Union{Symbol,Expr,QuoteNode})
    fn = Expr(:function, Expr(:tuple), var)
    node = gensym(var.value)
    push!(stmts, :($(esc(node)) = add_julia_node!(
        builder, $fn, inputs=[], name=$(QuoteNode(name)),
        typ=$(QuoteNode(typ)))))
    node
end

function parse_julia_expr!(stmts, bindings, name::Symbol, value,
                           typ::Union{Symbol,Expr,QuoteNode})
    fn = Expr(:function, Expr(:tuple), QuoteNode(value))
    node = gensym(repr(value))
    push!(stmts, :($(esc(node)) = add_julia_node!(
        builder, $fn, inputs=[], name=$(QuoteNode(name)),
        typ=$(QuoteNode(typ)))))
    node
end

function parse_julia_assignment!(stmts, bindings, line::Expr)
    if !MacroTools.@capture(line, lhs_ = rhs_) return false end
    (name::Symbol, typ) = parse_lhs(lhs)
    node = parse_julia_expr!(stmts, bindings, name, rhs, typ)
    bindings[name] = node
    true
end

split_addr!(keys, addr_expr::QuoteNode) = push!(keys, addr_expr)
split_addr!(keys, addr_expr::Symbol) = push!(keys, addr_expr)

function split_addr!(keys, addr_expr::Expr)
    @assert @capture(addr_expr, fst_ => snd_)
    push!(keys, fst)
    split_addr!(keys, snd)
end

choice_or_call_at(gen_fn::GenerativeFunction, addr_typ) = call_at(gen_fn, addr_typ)
choice_or_call_at(dist::Distribution, addr_typ) = choice_at(dist, addr_typ)

gen_arg_name(arg::Symbol) = gensym(arg)
gen_arg_name(arg::QuoteNode) = gensym(arg.value)
gen_arg_name(arg::Expr) = gensym(arg.head)
gen_arg_name(arg::Any) = gensym(repr(arg))

       # parse_trace_expr!(stmts::Vector{Expr}, bindings, name::Symbol("##XYZ"), line::Epr(not :(=)), typ)
function parse_trace_expr!(stmts, bindings, name, addr_expr, typ)
    # NOTE: typ is unused
    if !@capture(addr_expr, @macroname_(call_, site_)) return false end
    if macroname != STATIC_DSL_TRACE return false end
    if !isa(call, Expr) || call.head != :call
        return false
    end
    local addr::Symbol
    gen_fn_or_dist = gensym(call.args[1])
    push!(stmts, :($(esc(gen_fn_or_dist)) = $(esc(call.args[1]))))
    if isa(site, QuoteNode) && isa(site.value, Symbol)
        addr = site.value
        args = call.args[2:end]
    else
        # multi-part address syntactic sugar
        keys = []
        split_addr!(keys, site)
        if !isa(keys[1], QuoteNode) || !isa(keys[1].value, Symbol)
            return false
        end
        @assert length(keys) > 1
        addr = keys[1].value
        for key in keys[2:end]
            push!(stmts, :($(esc(gen_fn_or_dist)) = choice_or_call_at($(esc(gen_fn_or_dist)), Any)))
        end
        args = (call.args[2:end]..., reverse(keys[2:end])...)
    end
    node = gensym(name)
    if haskey(bindings, name)
        static_dsl_syntax_error(addr_expr, "Symbol $name already bound")
    end
    bindings[name] = node
    inputs = []
    for arg_expr in args
        arg_name = gen_arg_name(arg_expr)
        push!(inputs, parse_julia_expr!(stmts, bindings, arg_name, arg_expr, QuoteNode(Any)))
    end
    push!(stmts, :($(esc(node)) = add_addr_node!(
        builder, $(esc(gen_fn_or_dist)), inputs=[$(map(esc, inputs)...)], addr=$(QuoteNode(addr)),
        name=$(QuoteNode(name)))))
    true
end

function parse_trainable_param!(stmts::Vector{Expr}, bindings, line::Expr)
    if MacroTools.@capture(line, @macroname_ expr_)
        if macroname != STATIC_DSL_PARAM return false end
        (name::Symbol, typ) = parse_lhs(expr)
        if haskey(bindings, name)
            static_dsl_syntax_error(addr_expr, "Symbol $name already bound")
        end
        node = gensym(name)
        bindings[name] = node
        push!(stmts, :($(esc(node)) = add_trainable_param_node!(
            builder, $(QuoteNode(name)), typ=$(QuoteNode(typ)))))
        true
    else
        return false
    end
end

function parse_trace_line!(stmts::Vector{Expr}, bindings, line::Expr)
    if MacroTools.@capture(line, lhs_ = rhs_)
        (name::Symbol, typ) = parse_lhs(lhs)
        parse_trace_expr!(stmts, bindings, name, rhs, typ)
    else
        name = gensym()
        typ = QuoteNode(Any)
        parse_trace_expr!(stmts, bindings, name, line, typ)
    end
end

# return foo (must be a symbol) or return @trace(..)
function parse_return!(stmts::Vector{Expr}, bindings, line::Expr)
    if !MacroTools.@capture(line, return expr_) return false end
    if isa(expr, Expr) && expr.head == :macrocall
        var = gensym()
        typ = QuoteNode(Any)
        if !parse_trace_expr!(stmts, bindings, var, expr, typ)
            return false # the right-hand-side is a macro but not a valid `@trace` expression
        end
    elseif isa(expr, Symbol)
        var = expr
    else
        return false
    end
    if haskey(bindings, var)
        node = bindings[var]
        push!(stmts, :(set_return_node!(builder, $(esc(node)))))
        return true
    else
        error("Tried to return $var, which is not a locally bound variable")
    end
end

function parse_static_dsl_function_body!(stmts::Vector{Expr},
                                         bindings::Dict{Symbol,Symbol},
                                         expr)
    # TODO use line number nodes to provide better error messages in generated code
    if !isa(expr, Expr) || expr.head != :block
        static_dsl_syntax_error(expr)
    end
    for line in expr.args
        isa(line, LineNumberNode) && continue
        !isa(line, Expr) && static_dsl_syntax_error(line)

        # @param name::type
        parse_trainable_param!(stmts, bindings, line) && continue

        # lhs = @trace(rhs..) or @trace(rhs)
        parse_trace_line!(stmts, bindings, line) && continue

        # lhs = rhs
        # (only run if parsing as choice and call both fail)
        parse_julia_assignment!(stmts, bindings, line) && continue

        # return ..
        parse_return!(stmts, bindings, line) && continue

        static_dsl_syntax_error(line)
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
