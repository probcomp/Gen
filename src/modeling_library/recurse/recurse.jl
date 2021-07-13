struct Production{V,U}
    value::V
    children::Vector{U}
end

struct ProductionDiff <: Diff
    value_diff::Diff
    children_diff::VectorDiff
end

# the production kernel takes a value (of type U)
# and returns a value of type Tuple{V,Vector{U}}

# the reduction kernel takes an input value of tyoe Tuple{V,Vector{W}} and
# returns a value of type W. The Diff type for the Vector{W} is a VectorDiff.

using FunctionalCollections: assoc, dissoc, PersistentHashMap
using DataStructures: PriorityQueue, dequeue!, enqueue!

#################
# recurse trace #
#################

struct RecurseTrace{S,T,U,V,W,X,Y} <: Trace
    gen_fn::GenerativeFunction
    production_traces::PersistentHashMap{Int,S}
    aggregation_traces::PersistentHashMap{Int,T}
    max_branch::Int
    score::Float64
    root_idx::Int
    num_has_choices::Int
end

get_gen_fn(trace::RecurseTrace) = trace.gen_fn

function get_args(trace::RecurseTrace)
    root_arg = get_args(trace.production_traces[trace.root_idx])[1]
    (root_arg, trace.root_idx)
end

function get_retval(trace::RecurseTrace)
    get_retval(trace.aggregation_traces[trace.root_idx])
end

get_score(trace::RecurseTrace) = trace.score

# TODO assumes there is no untraced randomness
project(trace::RecurseTrace, ::EmptySelection) = 0.

##############################
# recurse assignment wrapper #
##############################

struct RecurseTraceChoiceMap <: ChoiceMap
    trace::RecurseTrace
end

get_choices(trace::RecurseTrace) = RecurseTraceChoiceMap(trace)
Base.getindex(trace::RecurseTrace, addr) = get_choices(trace)[addr]

function Base.isempty(choices::RecurseTraceChoiceMap)
    choices.trace.num_has_choices == 0
end

get_address_schema(::Type{RecurseTraceChoiceMap}) = DynamicAddressSchema()

function get_submap(choices::RecurseTraceChoiceMap,
                      addr::Tuple{Int,Val{:production}})
    idx = addr[1]
    if !haskey(choices.trace.aggregation_traces, idx)
        EmptyChoiceMap()
    else
        get_choices(choices.trace.production_traces[idx])
    end
end

function get_submap(choices::RecurseTraceChoiceMap,
                      addr::Tuple{Int,Val{:aggregation}})
    idx = addr[1]
    if !haskey(choices.trace.aggregation_traces, idx)
        EmptyChoiceMap()
    else
        get_choices(choices.trace.aggregation_traces[idx])
    end
end

function get_submap(choices::RecurseTraceChoiceMap, addr::Pair)
    _get_submap(choices, addr)
end

function has_value(choices::RecurseTraceChoiceMap, addr::Pair)
    _has_value(choices, addr)
end

function get_value(choices::RecurseTraceChoiceMap, addr::Pair)
    _get_value(choices, addr)
end

get_values_shallow(choices::RecurseTraceChoiceMap) = ()

function get_submaps_shallow(choices::RecurseTraceChoiceMap)
    production_iter = (((idx, Val(:production)), get_choices(subtrace))
        for (idx, subtrace) in choices.trace.production_traces)
    aggregation_iter = (((idx, Val(:aggregation)), get_choices(subtrace))
        for (idx, subtrace) in choices.trace.aggregation_traces)
    Iterators.flatten((production_iter, aggregation_iter))
end

# TODO when lightweight Gen functions properly declare their argument and return types, use:
# production_kern::GenerativeFunction{Tuple{V,Vector{U}},S}
# aggregation_kern::GenerativeFunction{W,T}

""""
    Recurse(production_kernel, aggregation_kernel, max_branch,
         ::Type{U}, ::Type{V}, ::Type{W})

Constructor for recurse production and aggregation function.
"""
struct Recurse{S,T,U,V,W,X,Y} <: GenerativeFunction{W,RecurseTrace{S,T,U,V,W,X,Y}}
    production_kern::GenerativeFunction{X,S}
    aggregation_kern::GenerativeFunction{Y,T}
    max_branch::Int
end

function Recurse(production_kernel::GenerativeFunction{X,S},
                 aggregation_kernel::GenerativeFunction{Y,T},
                 max_branch::Int, ::Type{U}, ::Type{V}, ::Type{W}) where {S,T,U,V,W,X,Y}
    Recurse{S,T,U,V,W,X,Y}(production_kernel, aggregation_kernel, max_branch)
end

# TODO
accepts_output_grad(::Recurse) = false

function (gen_fn::Recurse)(args...)
    (_, _, retval) = propose(gen_fn, args)
    retval
end

function get_child(parent::Int, child_num::Int, max_branch::Int)
    @assert child_num >= 1 && child_num <= max_branch
    (parent - 1) * max_branch + child_num + 1
end

function get_child_num(child::Int, max_branch::Int)
    @assert max_branch > 0
    (child - 2) % max_branch + 1
end

function get_parent(child::Int, max_branch::Int)
    @assert max_branch > 0
    child_num = get_child_num(child, max_branch)
    div(child - 1 - child_num, max_branch) + 1
end

function get_num_children(production_trace)
    length(get_retval(production_trace).children)
end

function get_production_input(gen_fn::Recurse{S,T,U,V,W}, cur::Int,
                              production_traces::AbstractDict{Int,S},
                              root_idx::Int, root_production_input::U) where {S,T,U,V,W}
    if cur == root_idx
        return root_production_input
    else
        parent = get_parent(cur, gen_fn.max_branch)
        child_num = get_child_num(cur, gen_fn.max_branch)
        # return type of parent is Tuple{V,Vector{U}}
        @assert haskey(production_traces, parent)
        return get_retval(production_traces[parent]).children[child_num]::U
    end
end

function get_aggregation_input(gen_fn::Recurse{S,T,U,V,W}, cur::Int,
                               production_traces::AbstractDict{Int,S},
                               aggregation_traces::AbstractDict{Int,T}) where {S,T,U,V,W}
    # requires that the corresponding production trace exists (for the v)
    # also requires that the child aggregation traces exist (for the w's)
    # does not require that this aggregation trace already exists
    vinput::V = get_retval(production_traces[cur]).value
    num_children = get_num_children(production_traces[cur])
    winputs::Vector{W} = [
        get_retval(aggregation_traces[get_child(cur, i, gen_fn.max_branch)])
        for i=1:num_children]
    return (vinput, winputs)
end


function get_production_constraints(constraints::ChoiceMap, cur::Int)
    get_submap(constraints, (cur, Val(:production)))
end

function get_aggregation_constraints(constraints::ChoiceMap, cur::Int)
    get_submap(constraints, (cur, Val(:aggregation)))
end

function to_serializable_trace(tr::RecurseTrace)
    return GenericSerializableTrace(
        (
            Dict(k => to_serializable_trace(subtr) for (k, subtr) in tr.production_traces),
            Dict(k => to_serializable_trace(subtr) for (k, subtr) in tr.aggregation_traces)
        ),
        (tr.max_branch, tr.score, tr.root_idx, tr.num_has_choices)
    )
end
function from_serializable_trace(st::GenericSerializableTrace, gf::Recurse{S, T}) where {S, T}
    production_traces = PersistentHashMap{Int, S}()
    for (k, subst) in st.subtraces[1]
        production_traces = assoc(production_traces, k, from_serializable_trace(subst, gf.production_kern))
    end
    aggregation_traces = PersistentHashMap{Int, T}()
    for (k, subst) in st.subtraces[2]
        aggregation_traces = assoc(aggregation_traces, k, from_serializable_trace(subst, gf.aggregation_kern))
    end
    return get_trace_type(gf)(gf, production_traces, aggregation_traces, st.properties...)
end

############
# simulate #
############

function simulate(gen_fn::Recurse{S,T,U,V,W,X,Y}, args::Tuple{U,Int}) where {S,T,U,V,W,X,Y}
    (root_production_input::U, root_idx::Int) = args
    production_traces = PersistentHashMap{Int,S}()
    aggregation_traces = PersistentHashMap{Int,T}()
    score = 0.
    num_has_choices = 0

    # production phase
    # does not matter in which order we visit (since children are inserted after parents)
    prod_to_visit = Set{Int}([root_idx])
    while !isempty(prod_to_visit)
        local subtrace::S
        local input::U
        cur = first(prod_to_visit)
        delete!(prod_to_visit, cur)
        input = get_production_input(gen_fn, cur, production_traces, root_idx, root_production_input)
        subtrace = simulate(gen_fn.production_kern, (input,))
        score += get_score(subtrace)
        production_traces = assoc(production_traces, cur, subtrace)
        children_inputs::Vector{U} = get_retval(subtrace).children
        for child_num in 1:length(children_inputs)
            push!(prod_to_visit, get_child(cur, child_num, gen_fn.max_branch))
        end
        if !isempty(get_choices(subtrace))
            num_has_choices += 1
        end
    end

    # aggregation phase
    # visit children first
    agg_to_visit = sort(collect(keys(production_traces)), rev=true)
    for cur in agg_to_visit
        local subtrace::T
        local input::Tuple{V,Vector{W}}
        input = get_aggregation_input(gen_fn, cur, production_traces, aggregation_traces)
        subtrace = simulate(gen_fn.aggregation_kern, input)
        score += get_score(subtrace)
        aggregation_traces = assoc(aggregation_traces, cur, subtrace)
        if !isempty(get_choices(subtrace))
            num_has_choices += 1
        end
    end

    RecurseTrace{S,T,U,V,W,X,Y}(gen_fn,
                production_traces, aggregation_traces, gen_fn.max_branch,
                score, root_idx, num_has_choices)
end

############
# generate #
############

function generate(gen_fn::Recurse{S,T,U,V,W,X,Y}, args::Tuple{U,Int},
                    constraints::ChoiceMap) where {S,T,U,V,W,X,Y}
    (root_production_input::U, root_idx::Int) = args
    production_traces = PersistentHashMap{Int,S}()
    aggregation_traces = PersistentHashMap{Int,T}()
    weight = 0.
    score = 0.
    num_has_choices = 0

    # production phase
    # does not matter in which order we visit (since children are inserted after parents)
    prod_to_visit = Set{Int}([root_idx])
    while !isempty(prod_to_visit)
        local subtrace::S
        local input::U
        cur = first(prod_to_visit)
        delete!(prod_to_visit, cur)
        input = get_production_input(gen_fn, cur, production_traces, root_idx, root_production_input)
        subconstraints = get_production_constraints(constraints, cur)
        (subtrace, subweight) = generate(gen_fn.production_kern, (input,), subconstraints)
        score += get_score(subtrace)
        production_traces = assoc(production_traces, cur, subtrace)
        weight += subweight
        children_inputs::Vector{U} = get_retval(subtrace).children
        for child_num in 1:length(children_inputs)
            push!(prod_to_visit, get_child(cur, child_num, gen_fn.max_branch))
        end
        if !isempty(get_choices(subtrace))
            num_has_choices += 1
        end
    end

    # aggregation phase
    # visit children first
    agg_to_visit = sort(collect(keys(production_traces)), rev=true)
    for cur in agg_to_visit
        local subtrace::T
        local input::Tuple{V,Vector{W}}
        input = get_aggregation_input(gen_fn, cur, production_traces, aggregation_traces)
        subconstraints = get_aggregation_constraints(constraints, cur)
        (subtrace, subweight) = generate(gen_fn.aggregation_kern, input, subconstraints)
        score += get_score(subtrace)
        aggregation_traces = assoc(aggregation_traces, cur, subtrace)
        weight += subweight
        if !isempty(get_choices(subtrace))
            num_has_choices += 1
        end
    end

    trace = RecurseTrace{S,T,U,V,W,X,Y}(gen_fn,
                production_traces, aggregation_traces, gen_fn.max_branch,
                score, root_idx, num_has_choices)
    return (trace, weight)
end


##########
# update #
##########

struct NodeQueue{T}
    ordering::T
    pq::PriorityQueue{Int,Int}
end

function NodeQueue(ordering)
    pq = PriorityQueue{Int,Int}(ordering)
    NodeQueue(ordering, pq)
end

function Base.push!(queue::NodeQueue, node::Int)
    if !haskey(queue.pq, node)
        enqueue!(queue.pq, node, node)
    end
end
Base.pop!(queue::NodeQueue) = dequeue!(queue.pq)
Base.in(node::Int, queue::NodeQueue) = haskey(queue.pq, node)
Base.isempty(queue::NodeQueue) = isempty(queue.pq)


function recurse_unpack_constraints(constraints::ChoiceMap)
    production_constraints = Dict{Int, Any}()
    aggregation_constraints = Dict{Int, Any}()
    for (addr, node) in get_submaps_shallow(constraints)
        idx::Int = addr[1]
        if addr[2] == Val(:production)
            production_constraints[idx] = node
        elseif addr[2] == Val(:aggregation)
            aggregation_constraints[idx] = node
        else
            error("Unknown address: $addr")
        end
    end
    if !isempty(get_values_shallow(constraints))
        error("Unknown address: $(first(get_values_shallow(constraints))[1])")
    end
    return (production_constraints, aggregation_constraints)
end

function dissoc_subtree!(discard::DynamicChoiceMap,
						 production_traces::PersistentHashMap{Int,S},
                         aggregation_traces::PersistentHashMap{Int,T},
                         root::Int, max_branch::Int) where {S,T}
    num_has_choices = 0
    production_subtrace = production_traces[root]
    aggregation_subtrace = aggregation_traces[root]
	set_submap!(discard, (root, Val(:production)), get_choices(production_subtrace))
	set_submap!(discard, (root, Val(:aggregation)), get_choices(aggregation_subtrace))
    if !isempty(get_choices(production_subtrace))
        num_has_choices += 1
    end
    if !isempty(get_choices(aggregation_subtrace))
        num_has_choices += 1
    end
    num_children = get_num_children(production_subtrace)
    removed_score = get_score(production_subtrace) + get_score(aggregation_subtrace)
    for child_num=1:num_children
        child = get_child(root, child_num, max_branch)
        (production_traces, aggregation_traces, child_removed_score, child_num_has_choices) = dissoc_subtree!(
            discard, production_traces, aggregation_traces, child, max_branch)
        removed_score += child_removed_score
        num_has_choices += child_num_has_choices
    end
    production_traces = dissoc(production_traces, root)
    aggregation_traces = dissoc(aggregation_traces, root)
    return (production_traces, aggregation_traces, removed_score, num_has_choices)
end

function get_production_argdiff(production_retdiffs::Dict{Int,Diff},
                                root_idx::Int, root_argdiff, cur::Int,
                                max_branch::Int)
    if cur == root_idx
        return root_argdiff
    else
        parent = get_parent(cur, max_branch)
        if !haskey(production_retdiffs, parent)
            return NoChange()
        else
            retdiff = production_retdiffs[parent]
            @assert retdiff != NoChange()
            child_num = get_child_num(cur, max_branch)
            if isa(retdiff, ProductionDiff)
                # TODO: test this code path
                children_diff = production_retdiffs[parent].children_diff
                if child_num > children_diff.prev_len
                    return UnknownChange()
                else
                    if haskey(children_diff.updated, child_num)
                        return children_diff.updated[child_num]
                    else
                        return NoChange()
                    end
                end
            else
                return UnknownChange()
            end
        end
    end
end

function get_aggregation_argdiffs(production_retdiffs::Dict{Int,Diff},
                                  aggregation_retdiffs::Dict{Int,Diff},
                                  idx_to_prev_num_children::Dict{Int,Int},
                                  production_traces, cur::Int)
    if haskey(production_retdiffs, cur)
        production_retdiff = production_retdiffs[cur]
        if isa(production_retdiff, ProductionDiff)
            dv = production_retdiff.value_diff
        else
            dv = UnknownChange()
        end
    else
        dv = NoChange()
    end
    new_num_children = get_num_children(production_traces[cur])
    if haskey(idx_to_prev_num_children, cur)
        prev_num_children = idx_to_prev_num_children[cur]
    else
        prev_num_children = new_num_children
    end
    dws = Dict{Int,Diff}()
    for child_num=1:min(prev_num_children,new_num_children)
        if haskey(aggregation_retdiffs, child_num)
            dws[child_num] = aggregation_retdiffs[child_num]
        end
    end
    (dv, VectorDiff(new_num_children, prev_num_children, dws))
end

function update(trace::RecurseTrace{S,T,U,V,W,X,Y},
                new_args::Tuple{U,Int},
                argdiffs::Tuple,
                constraints::ChoiceMap) where {S,T,U,V,W,X,Y}
    gen_fn = get_gen_fn(trace)
    (root_production_input::U, root_idx::Int) = new_args
    if root_idx != get_args(trace)[2]
        error("Cannot change root_idx argument")
    end

    (root_argdiff::Diff,) = argdiffs

    production_traces = trace.production_traces
    aggregation_traces = trace.aggregation_traces
    (production_constraints, aggregation_constraints) = recurse_unpack_constraints(constraints)
    discard = choicemap()

    # initial score from previous trace
    score = trace.score
    weight = 0.
    num_has_choices = trace.num_has_choices

    # initialize set of production nodes to visit
    # visit nodes in forward order (starting with root)
    prod_to_visit = NodeQueue(Base.Order.Forward)
    for node in keys(production_constraints)
        push!(prod_to_visit, node)
    end

    # if the root argdiff is not noargdiff then we visit the root production node
    if root_argdiff != NoChange() && !(root_idx in prod_to_visit)
        push!(prod_to_visit, root_idx)
    end

    # initialize set of aggregation nodes to visit
    # visit nodes in reverse order (starting with leaves)
    agg_to_visit = NodeQueue(Base.Order.Reverse)
    for node in keys(aggregation_constraints)
        push!(agg_to_visit, node)
    end

    # only store for nodes that are retained
    # if a value is not present for a retained node, then the number of children of that node did not change
    idx_to_prev_num_children = Dict{Int,Int}()

    # production phase
    production_retdiffs = Dict{Int,Diff}() # elements with no difference are ommitted
    while !isempty(prod_to_visit)
        local subtrace::S

        cur = pop!(prod_to_visit)
        subconstraints = get_production_constraints(constraints, cur)
        input = (get_production_input(gen_fn, cur, production_traces, root_idx, root_production_input)::U,)

        if haskey(production_traces, cur)
            # the node exists already
            local subargdiff
            local subretdiff::Diff

            # get argdiff for this production node
            subargdiff = get_production_argdiff(production_retdiffs, root_idx,
                                                root_argdiff, cur, gen_fn.max_branch)

            # call update on production kernel
            prev_subtrace = production_traces[cur]
            (subtrace, subweight, subretdiff, subdiscard) = update(
                prev_subtrace, input, (subargdiff,), subconstraints)
            prev_num_children = get_num_children(production_traces[cur])
            new_num_children = length(get_retval(subtrace).children)
            idx_to_prev_num_children[cur] = prev_num_children
            set_submap!(discard, (cur, Val(:production)), subdiscard)
            production_retdiffs[cur] = subretdiff

            # update trace, weight, and score
            production_traces = assoc(production_traces, cur, subtrace)
            weight += subweight
            score += subweight

            # update num_has_choices
            if !isempty(get_choices(prev_subtrace)) && isempty(get_choices(subtrace))
                num_has_choices -= 1
            elseif isempty(get_choices(prev_subtrace)) && !isempty(get_choices(subtrace))
                num_has_choices += 1
            end

            # delete children (and their descendants), both production and aggregation nodes
            for child_num=new_num_children+1:prev_num_children
                child = get_child(cur, child_num, gen_fn.max_branch)
                set_submap!(discard, (child, Val(:production)), get_choices(production_traces[child]))
                set_submap!(discard, (child, Val(:aggregation)), get_choices(aggregation_traces[child]))
                (production_traces, aggregation_traces, removed_score, removed_num_has_choices) = dissoc_subtree!(
                    discard, production_traces, aggregation_traces, child, gen_fn.max_branch)
                score -= removed_score
                num_has_choices -= removed_num_has_choices
                weight -= removed_score
            end

            # mark new children for processing
            for child_num=prev_num_children+1:new_num_children
                child = get_child(cur, child_num, gen_fn.max_branch)
                push!(prod_to_visit, child)
            end

            if isa(subretdiff, ProductionDiff)

                # maybe mark existing children for processing, if they have a custom retdiff
                # (otherwise they do not change)
                for child_num in keys(subretdiff.children_diff.updated)
                    @assert child_num <= min(prev_num_children,new_num_children)
                    child = get_child(cur, child_num, gen_fn.max_branch)
                    push!(prod_to_visit, child)
                end

                # mark corresponding aggregation node for processing if v has
                # changed, or if the number of children has changed
                if subretdiff.value_diff != NoChange() || prev_num_children != new_num_children
                    push!(agg_to_visit, cur)
                end
            else
                # treat it like UnknownDiff
                for child_num=1:min(prev_num_children,new_num_children)
                    child = get_child(cur, child_num, gen_fn.max_branch)
                    push!(prod_to_visit, child)
                end
                push!(agg_to_visit, cur)
            end

        else
            # the node does not exist already (and none of its children exist either)
            (subtrace, ) = generate(gen_fn.production_kern, input, subconstraints)

            # update trace, weight, and score
            production_traces = assoc(production_traces, cur, subtrace)
            weight += get_score(subtrace)
            score += get_score(subtrace)

            # update num_has_choices
            if !isempty(get_choices(subtrace))
                num_has_choices += 1
            end

            # mark corresponding aggregation node for processsing
            push!(agg_to_visit, cur)

            # mark children (which are all new) for processing
            for child_num=1:get_num_children(subtrace)
                child = get_child(cur, child_num, gen_fn.max_branch)
                push!(prod_to_visit, child)
            end
        end
    end

    # aggregation phase

    # reasons to visit an aggregation node:
    # - it is new
    # - it is directly constrained
    # - its corresponding production node has !isnodiff(dv)
    # - one or more of of its children has a !isnodiff(dw)
    # note: we already deleted aggregation nodes that are to be deleted above

    aggregation_retdiffs = Dict{Int,Diff}() # NoChange()s are ommitted
    local subtrace::T
    local subargdiffs::Tuple # (diff for V, diff for vector of Ws)
    local subretdiff # Diff
    local retdiff
    retdiff = NoChange()
    while !isempty(agg_to_visit)

        cur = pop!(agg_to_visit)
        subconstraints = get_aggregation_constraints(constraints, cur)
        input::Tuple{V,Vector{W}} = get_aggregation_input(
            gen_fn, cur, production_traces, aggregation_traces)

        # if the node exists already
        if haskey(aggregation_traces, cur)

            # get argdiff for this aggregation node
            subargdiffs = get_aggregation_argdiffs(production_retdiffs, aggregation_retdiffs,
                                                   idx_to_prev_num_children, production_traces, cur)

            # call update on aggregation kernel
            prev_subtrace = aggregation_traces[cur]
            (subtrace, subweight, subretdiff, subdiscard) = update(
                prev_subtrace, input, subargdiffs, subconstraints)

            # update trace, weight, and score, and discard
            aggregation_traces = assoc(aggregation_traces, cur, subtrace)
            weight += subweight
            score += subweight
            set_submap!(discard, (cur, Val(:aggregation)), subdiscard)

            # update num_has_choices
            if !isempty(get_choices(prev_subtrace)) && isempty(get_choices(subtrace))
                num_has_choices -= 1
            elseif isempty(get_choices(prev_subtrace)) && !isempty(get_choices(subtrace))
                num_has_choices += 1
            end

            # take action based on our subretdiff
            if cur == root_idx
                retdiff = subretdiff
            else
                if subretdiff != NoChange()
                    aggregation_retdiffs[cur] = subretdiff
                    push!(agg_to_visit, get_parent(cur, gen_fn.max_branch))
                end
            end

        # if the node does not exist (but its children do, since we created them already)
        else
            (subtrace, _) = generate(gen_fn.aggregation_kern, input, subconstraints)

            # update trace, weight, and score
            aggregation_traces = assoc(aggregation_traces, cur, subtrace)
            weight += get_score(subtrace)
            score += get_score(subtrace)

            # update num_has_choices
            if !isempty(get_choices(subtrace))
                num_has_choices += 1
            end

            # the parent should have been marked during production phase
            @assert cur != 1
            @assert get_parent(cur, gen_fn.max_branch) in agg_to_visit
        end

    end

    new_trace = RecurseTrace{S,T,U,V,W,X,Y}(gen_fn,
        production_traces, aggregation_traces, gen_fn.max_branch,
        score, root_idx, num_has_choices)

    return (new_trace, weight, retdiff, discard)
end

export Recurse
export Production, ProductionDiff
export get_child
