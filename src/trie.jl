##################
# Trie #
##################

struct Trie{K,V}
    leaf_nodes::Dict{K,V}
    internal_nodes::Dict{K,Trie{K,V}}
end

Trie{K,V}() where {K,V} = Trie(Dict{K,V}(), Dict{K,Trie{K,V}}())

# copy constructor for something supported read-only trie interface
function Trie(other)
    trie = Trie{Any,Any}()
    for (key, value) in get_leaf_nodes(other)
        set_leaf_node!(trie, key, value)
    end
    for (key, node) in get_internal_nodes(other)
        sub_trie = Trie(node)
        set_internal_node!(trie, key, sub_trie)
    end
    trie
end

# TODO come up with a better printing method (and something nice for Jupyter
# notebooks)
import JSON
Base.println(trie::Trie) = JSON.print(trie, 4)

# invariant: all internal nodes are nonempty
Base.isempty(trie::Trie) = isempty(trie.leaf_nodes) && isempty(trie.internal_nodes)
get_leaf_nodes(trie::Trie) = trie.leaf_nodes
get_internal_nodes(trie::Trie) = trie.internal_nodes

Base.:(==)(t1::Trie, t2::Trie) = get_leaf_nodes(t1) == get_leaf_nodes(t2) && get_internal_nodes(t1) == get_internal_nodes(t2)
function Base.values(trie::Trie)
    iterators = convert(Vector{Any}, collect(map(values, values(trie.internal_nodes))))
    push!(iterators, values(trie.leaf_nodes))
    Iterators.flatten(iterators)
end

function has_internal_node(trie::Trie, addr)
    haskey(trie.internal_nodes, addr)
end

function has_internal_node(trie::Trie, addr::Pair)
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        has_internal_node(trie.internal_nodes[first], rest)
    else
        false
    end
end

function get_internal_node(trie::Trie, addr)
    trie.internal_nodes[addr]
end

function get_internal_node(trie::Trie, addr::Pair)
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        get_internal_node(trie.internal_nodes[first], rest)
    else
        throw(KeyError(trie, addr))
    end
end

function set_internal_node!(trie::Trie{K,V}, addr, new_node::Trie{K,V}) where {K,V}
    if !isempty(new_node)
        trie.internal_nodes[addr] = new_node
    end
end

function set_internal_node!(trie::Trie{K,V}, addr::Pair, new_node::Trie{K,V}) where {K,V}
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        node = trie.internal_nodes[first]
    else
        node = Trie{K,V}()
        trie.internal_nodes[first] = node
    end
    set_internal_node!(node, rest, new_node)
end

function delete_internal_node!(trie::Trie, addr)
    delete!(trie.internal_nodes, addr)
    return isempty(trie.leaf_nodes) && isempty(trie.internal_nodes)
end

function delete_internal_node!(trie::Trie, addr::Pair)
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        node = trie.internal_nodes[first]
        if delete_internal_node!(node, rest, new_node)
            delete!(trie.internal_nodes, first)
        end
    end
    return isempty(trie.leaf_nodes) && isempty(trie.internal_nodes)
end

function has_leaf_node(trie::Trie, addr)
    haskey(trie.leaf_nodes, addr)
end

function has_leaf_node(trie::Trie, addr::Pair)
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        has_leaf_node(trie.internal_nodes[first], rest)
    else
        false
    end
end

function get_leaf_node(trie::Trie, addr)
    trie.leaf_nodes[addr]
end

function get_leaf_node(trie::Trie, addr::Pair)
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        get_leaf_node(trie.internal_nodes[first], rest)
    else
        throw(KeyError(trie, addr))
    end
end

function set_leaf_node!(trie::Trie, addr, value)
    trie.leaf_nodes[addr] = value
end

function set_leaf_node!(trie::Trie{K,V}, addr::Pair, value) where {K,V}
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        node = trie.internal_nodes[first]
    else
        node = Trie{K,V}()
        trie.internal_nodes[first] = node
    end
    node = trie.internal_nodes[first]
    set_leaf_node!(node, rest, value)
end

function delete_leaf_node!(trie::Trie, addr)
    delete!(trie.leaf_nodes, addr)
    return isempty(trie.leaf_nodes) && isempty(trie.internal_nodes)
end

function delete_leaf_node!(trie::Trie, addr::Pair)
    (first, rest) = addr
    if haskey(trie.internal_nodes[first])
        node = trie.internal_nodes[first]
        if delete_leaf_node!(node, rest)
            delete!(trie.internal_nodes, first)
        end
    end
    return isempty(trie.leaf_nodes) && isempty(trie.internal_nodes)
end

Base.setindex!(trie::Trie, value, addr) = set_leaf_node!(trie, addr, value)

function Base.merge!(a::Trie{K,V}, b::Trie{K,V}) where {K,V}
    merge!(a.leaf_nodes, b.leaf_nodes)
    for (key, a_sub) in a.sub
        if haskey(b.sub, key)
            b_sub = b.sub[key]
            merge!(a_sub, b_sub)
        end
    end
    for (key, b_sub) in b.sub
        if !haskey(a.sub, key)
            a.sub[key] = b_sub
        end
    end
    a
end

get_address_schema(::Trie) = DynamicSchema()

Base.haskey(trie::Trie, key) = has_leaf_node(trie, key)

Base.getindex(trie::Trie, key) = get_leaf_node(trie, key)

"""
    triemap(trie::Trie, key_converter, leaf_converter)

Get a new trie by applying the function `key_converter` to every key in the trie
and applying the function `leaf_converter` to every leaf node in the trie.
"""
function triemap(trie::Trie{K, V}, key_converter, leaf_converter) where {K, V}
    new_keytype = Core.Compiler.return_type(key_converter, Tuple{K})
    KT = Union{Base.return_types(key_converter, (K,))...}
    LT = Union{Base.return_types(leaf_converter, (V,))...}
    converted_leafs = Dict{KT, LT}(
        key_converter(k) => leaf_converter(v) for (k, v) in trie.leaf_nodes
    )
    converted_internals = Dict{KT, Trie{KT, LT}}(
        key_converter(k) => convert_to_serializable_trie(subtrie, key_converter, leaf_converter, new_keytype, new_leaftype)
        for (k, subtrie) in trie.internal_nodes
    )
    
    return Trie{KT, LT}(converted_leafs, converted_internals)
end


export Trie
export set_internal_node!
export delete_internal_node!
export set_leaf_node!
export delete_leaf_node!
