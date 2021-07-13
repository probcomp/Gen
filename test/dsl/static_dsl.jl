#######################
# static DSL function #
#######################

using FunctionalCollections: PersistentVector

struct Params
    prob_outlier::Float64
    slope::Float64
    intercept::Float64
    inlier_std::Float64
    outlier_std::Float64
end

@gen (static) function datum(x::Float64, (grad)(params::Params))
    is_outlier = @trace(bernoulli(params.prob_outlier), :z)
    std = is_outlier ? params.outlier_std : params.inlier_std
    y = @trace(normal(x * params.slope + params.intercept, std), :y)
    return y
end

data_fn = Map(datum)

"""
my documentation
"""
@gen (static) function model(xs::Vector{Float64})
    n = length(xs)
    inlier_std = @trace(gamma(1, 1), :inlier_std)
    outlier_std = @trace(gamma(1, 1), :outlier_std)
    slope = @trace(normal(0, 2), :slope)
    intercept = @trace(normal(0, 2), :intercept)
    params::Params = Params(0.5, slope, intercept, inlier_std, outlier_std)
    ys = @trace(data_fn(xs, fill(params, n)), :data)
    return ys
end


@gen (static) function at_choice_example_1(i::Int)
    ret = @trace(bernoulli(0.5), :x => i)
end

# @trace(choice_at(bernoulli)(0.5, i), :x)

@gen (static) function at_choice_example_2(i::Int)
    ret = @trace(bernoulli(0.5), :x => i => :y)
end

# @trace(call_at(choice_at(bernoulli))(0.5, i, :y), :x)

@gen function foo(mu)
    @trace(normal(mu, 1), :y)
end


@gen (static) function at_call_example_1(i::Int)
    mu = 1.123
    ret = @trace(foo(mu), :x => i)
end


@gen (static) function at_call_example_2(i::Int)
    mu = 1.123
    ret = @trace(foo(mu), :x => i => :y)
end

# Modules to test load_generated_functions
module MyModuleA
using Gen
@gen (static) function foo(mu)
    y = @trace(normal(mu, 1), :y)
    return y
end
load_generated_functions(@__MODULE__)
end

module MyModuleB
using Gen
@gen (static) function foo(mu)
    y = @trace(normal(mu, 1), :y)
    return y
end
@load_generated_functions
end

@testset "static DSL" begin

function get_node_by_name(ir, name::Symbol)
    nodes = filter((node) -> (node.name == name), ir.nodes)
    @assert length(nodes) == 1
    nodes[1]
end

function get_node_by_addr(ir, addr::Symbol)
    nodes = filter(n -> hasfield(typeof(n), :addr) && (n.addr==addr), ir.nodes)
    @assert length(nodes) == 1
    nodes[1]
end

@testset "check IR" begin

#####################
# check IR of datum #
#####################


ir = Gen.get_ir(typeof(datum))
# argument nodes
@test length(ir.arg_nodes) == 2
x = ir.arg_nodes[1]
params = ir.arg_nodes[2]
@test x.name == :x
@test x.typ == :Float64
@test !x.compute_grad
@test params.name == :params
@test params.typ == :Params
@test params.compute_grad

# choice nodes and call nodes
@test length(ir.choice_nodes) == 2
@test length(ir.call_nodes) == 0

# is_outlier
is_outlier = ir.choice_nodes[1]
@test is_outlier.addr == :z
@test is_outlier.typ == QuoteNode(Bool)
@test is_outlier.dist == bernoulli
@test length(is_outlier.inputs) == 1

# std
std = get_node_by_name(ir, :std)
@test isa(std, Gen.JuliaNode)
@test std.name == :std
@test std.typ == QuoteNode(Any)
@test length(std.inputs) == 2
in1 = std.inputs[1]
in2 = std.inputs[2]
@test (in1 === is_outlier && in2 === params) || (in2 === is_outlier && in1 === params)

# y
y = ir.choice_nodes[2]
@test y.addr == :y
@test y.typ == QuoteNode(Float64)
@test y.dist == normal
@test length(y.inputs) == 2
@test y.inputs[2] === std

# y_mean
y_mean = y.inputs[1]
@test isa(y_mean, Gen.JuliaNode)
@test y_mean.typ == QuoteNode(Any)
@test length(y_mean.inputs) == 2
in1 = y_mean.inputs[1]
in2 = y_mean.inputs[2]
@test (in1 === x && in2 === params) || (in2 === x && in1 === params)

# prob outlier
prob_outlier = is_outlier.inputs[1]
@test isa(prob_outlier, Gen.JuliaNode)
@test length(prob_outlier.inputs) == 1
@test prob_outlier.inputs[1] === params
@test prob_outlier.typ == QuoteNode(Any)

@test ir.return_node === y

#####################
# check IR of model #
#####################

ir = Gen.get_ir(typeof(model))
@test length(ir.arg_nodes) == 1
xs = ir.arg_nodes[1]
@test xs.name == :xs
@test xs.typ == :(Vector{Float64})
@test !xs.compute_grad

# choice nodes and call nodes
@test length(ir.choice_nodes) == 4
@test length(ir.call_nodes) == 1

# inlier_std
inlier_std = ir.choice_nodes[1]
@test inlier_std.addr == :inlier_std
@test inlier_std.typ == QuoteNode(Float64)
@test inlier_std.dist == gamma
@test length(inlier_std.inputs) == 2

# outlier_std
outlier_std = ir.choice_nodes[2]
@test outlier_std.addr == :outlier_std
@test outlier_std.typ == QuoteNode(Float64)
@test outlier_std.dist == gamma
@test length(outlier_std.inputs) == 2

# slope
slope = ir.choice_nodes[3]
@test slope.addr == :slope
@test slope.typ == QuoteNode(Float64)
@test slope.dist == normal
@test length(slope.inputs) == 2

# intercept
intercept = ir.choice_nodes[4]
@test intercept.addr == :intercept
@test intercept.typ == QuoteNode(Float64)
@test intercept.dist == normal
@test length(intercept.inputs) == 2

# data
ys = ir.call_nodes[1]
@test ys.addr == :data
@test ys.typ == QuoteNode(PersistentVector{Float64})
@test ys.generative_function == data_fn
@test length(ys.inputs) == 2
@test ys.inputs[1] == xs

# params
params = get_node_by_name(ir, :params)
@test isa(params, Gen.JuliaNode)
@test params.name == :params
@test params.typ == :Params
@test length(params.inputs) == 4
@test slope in params.inputs
@test intercept in params.inputs
@test inlier_std in params.inputs
@test outlier_std in params.inputs

# n
n = get_node_by_name(ir, :n)
@test isa(n, Gen.JuliaNode)
@test n.name == :n
@test n.typ == QuoteNode(Any)
@test length(n.inputs) == 1
@test n.inputs[1] === xs

# params_vec
params_vec = ys.inputs[2]
@test isa(params_vec, Gen.JuliaNode)
@test params_vec.typ == QuoteNode(Any)
@test length(params_vec.inputs) == 2
in1 = params_vec.inputs[1]
in2 = params_vec.inputs[2]
@test (in1 === params && in2 === n) || (in2 === params && in1 === n)

@test ir.return_node === ys
end

@testset "at_choice" begin

# at_choice_example_1
ir = Gen.get_ir(typeof(at_choice_example_1))
i = get_node_by_name(ir, :i)
ret = get_node_by_addr(ir, :x)
@test isa(ret, Gen.GenerativeFunctionCallNode)
@test ret.addr == :x
@test length(ret.inputs) == 2
@test isa(ret.inputs[1], Gen.JuliaNode) # () -> 0.5
@test ret.inputs[2] === i
at = ret.generative_function
@test isa(at, Gen.ChoiceAtCombinator)
@test at.dist == bernoulli

# at_choice_example_2
ir = Gen.get_ir(typeof(at_choice_example_2))
i = get_node_by_name(ir, :i)
ret = get_node_by_addr(ir, :x)
@test isa(ret, Gen.GenerativeFunctionCallNode)
@test ret.addr == :x
@test length(ret.inputs) == 3
@test isa(ret.inputs[1], Gen.JuliaNode) # () -> 0.5
@test isa(ret.inputs[2], Gen.JuliaNode) # () -> :y
@test ret.inputs[3] === i
at = ret.generative_function
@test isa(at, Gen.CallAtCombinator)
at2 = at.kernel
@test isa(at2, Gen.ChoiceAtCombinator)
@test at2.dist == bernoulli
end


@testset "at_call" begin

# at_call_example_1
ir = Gen.get_ir(typeof(at_call_example_1))
i = get_node_by_name(ir, :i)
ret = get_node_by_addr(ir, :x)
@test isa(ret, Gen.GenerativeFunctionCallNode)
@test ret.addr == :x
@test length(ret.inputs) == 2
@test isa(ret.inputs[1], Gen.JuliaNode) # () -> 0.5
@test ret.inputs[2] === i
at = ret.generative_function
@test isa(at, Gen.CallAtCombinator)
@test at.kernel == foo

#at_call_example_2
ir = Gen.get_ir(typeof(at_call_example_2))
i = get_node_by_name(ir, :i)
ret = get_node_by_addr(ir, :x)
@test isa(ret, Gen.GenerativeFunctionCallNode)
@test ret.addr == :x
@test length(ret.inputs) == 3
@test isa(ret.inputs[1], Gen.JuliaNode) # () -> 0.5
@test isa(ret.inputs[1], Gen.JuliaNode) # () -> 0.5
@test isa(ret.inputs[2], Gen.JuliaNode) # () -> :y
@test ret.inputs[3] === i
at = ret.generative_function
@test isa(at, Gen.CallAtCombinator)
at2 = at.kernel
@test isa(at2, Gen.CallAtCombinator)
@test at2.kernel == foo
end


@testset "tracked diff annotation" begin

@gen (static, diffs) function bar(x)
    return x
end

@test Gen.get_options(typeof(bar)).track_diffs

@gen (static) function bar(x)
    return x
end

@test !Gen.get_options(typeof(bar)).track_diffs

end

@testset "no julia cache annotation" begin

@gen (static, nojuliacache) function bar(x)
    return x
end

@test !Gen.get_options(typeof(bar)).cache_julia_nodes

@gen (static) function bar(x)
    return x
end

@test Gen.get_options(typeof(bar)).cache_julia_nodes

end



@testset "trainable parameters" begin

@gen (static) function foo()
    @param theta::Float64
    return theta
end

ir = Gen.get_ir(typeof(foo))
theta = get_node_by_name(ir, :theta)
@test isa(theta, Gen.TrainableParameterNode)
@test ir.return_node === theta

end

@testset "returning a trace directly" begin

@gen (static) function f1()
    x = @trace(normal(0, 1), :foo)
    return x
end

@gen (static) function f2()
    return @trace(normal(0, 1), :foo)
end

ir1 = Gen.get_ir(typeof(f1))
ir2 = Gen.get_ir(typeof(f2))
return_node1 = ir1.return_node
return_node2 = ir2.return_node
@test isa(return_node2, typeof(return_node1))
@test return_node2.dist == return_node1.dist

inputs1 = return_node1.inputs
inputs2 = return_node2.inputs
@test 0 == inputs1[1].fn() == inputs2[1].fn()
@test 1 == inputs1[2].fn() == inputs2[2].fn()

@test return_node2.name != return_node1.name
@test return_node2.addr == return_node1.addr
@test return_node2.typ === return_node1.typ

end

@testset "@trace within Julia expressions" begin

@gen (static) function circle(r, t, std)
    pt = (@trace(normal(r*cos(t), std), :x), @trace(normal(r*sin(t), std), :y))
    return pt
end

@gen (static) function beta_bernoulli()
    return @trace(bernoulli(@trace(beta(1, 1), :z)), :x)
end

load_generated_functions()

x, y = circle(2, pi/2, 0)
@test isapprox(x, 0.0, atol=1e-8) && isapprox(y, 2.0, atol=1e-8)
tr, w = generate(circle, (1, 0, 1), choicemap(:x => 1, :y => 1))
@test isapprox(w, logpdf(normal, 1, 1, 1) + logpdf(normal, 1, 0, 1))

tr, w = generate(beta_bernoulli, (), choicemap(:z => 1.0))
@test tr[:x] == get_retval(tr) == true
tr, w = generate(beta_bernoulli, (), choicemap(:z => 0.0))
@test tr[:x] == get_retval(tr) == false
tr, w = generate(beta_bernoulli, (), choicemap(:z => 0.5, :x => true))
@test isapprox(w, logpdf(beta, 0.5, 1, 1) + logpdf(bernoulli, true, 0.5))

end

@testset "repeated assignments" begin

@gen (static) function foo(x)
    x = x + 1
    return x
end

@gen (static) function bar(x)
    y = x + 1
    y = y + 1
    return y + 1
end

load_generated_functions()

ir = Gen.get_ir(typeof(foo))
x1 = get_node_by_name(ir, :x)
x2 = ir.return_node
@test isa(x1, Gen.ArgumentNode)
@test isa(x2, Gen.JuliaNode)
@test x1 !== x2
@test x2.inputs[1] === x1
@test foo(0) == 1

ir = Gen.get_ir(typeof(bar))
y1 = get_node_by_name(ir, :y)
y2 = ir.return_node.inputs[1]
@test y1 !== y2
@test y2.inputs[1] === y1
@test bar(0) == 3

end

@testset "tuple assignments" begin

@gen (static) function foo(params::Tuple)
    p, (mu1, sigma1), (mu2, sigma2) = params
    mu, sigma = @trace(bernoulli(p), :x) ? (mu1, sigma1) : (mu2, sigma2)
    return @trace(normal(mu, sigma), :y)
end

ir = Gen.get_ir(typeof(foo))
params = get_node_by_name(ir, :params)
p = get_node_by_name(ir, :p)
mu1 = get_node_by_name(ir, :mu1)
sigma2 = get_node_by_name(ir, :sigma2)
@test p.inputs[1] === params
@test mu1.inputs[1].inputs[1] === params
@test sigma2.inputs[1].inputs[1] === params

return_node = ir.return_node
inputs = return_node.inputs
@test inputs[1] === get_node_by_name(ir, :mu)
@test inputs[2] === get_node_by_name(ir, :sigma)
@test inputs[1].fn((:fst, :snd)) == :fst
@test inputs[2].fn((:fst, :snd)) == :snd

load_generated_functions()

params = (0.5, (0, 1), (1, 2))
tr, w = generate(foo, (params,), choicemap(:x => true, :y => 1))
@test get_retval(tr) == 1
@test isapprox(w, logpdf(bernoulli, true, 0.5) + logpdf(normal, 1, 0, 1))

tr, w = generate(foo, (params,), choicemap(:x => false, :y => 1))
@test get_retval(tr) == 1
@test isapprox(w, logpdf(bernoulli, false, 0.5) + logpdf(normal, 1, 1, 2))

end

@testset "use of 'end'" begin

@gen (static) function foo()
    x = [1, 2, 3, 4, 5, 6]
    y = x[3:end]
    return y
end

load_generated_functions()

@test foo() == [3, 4, 5, 6]

end

@testset "'return nothing'" begin

@gen (static) function foo()
    return nothing
end

load_generated_functions()

@test foo() == nothing

end

@testset "getindex(trace)" begin

@gen (static) function bar(r)
    a = @trace(normal(0, 1), :a)
    return r
end
@gen (static) function foo()
    x = @trace(bar(1), :x)
    yz = @trace(bar(2), :y => :z)
    u = @trace(normal(0, 1), :u)
    vw = @trace(normal(0, 1), :v => :w)
    ret = 7
    return ret
end

Gen.load_generated_functions()

constraints = choicemap()
constraints[:u] = 1.1
constraints[:v => :w] = 1.2
constraints[:x => :a] = 1.3
constraints[:y => :z => :a] = 1.4
trace, = generate(foo, (), constraints)

# random choices
@test trace[:u] == 1.1
@test trace[:v => :w] == 1.2
@test trace[:x => :a] == 1.3
@test trace[:y => :z => :a] == 1.4

# auxiliary state
@test trace[:x] == 1
@test trace[:y => :z] == 2

# return value
@test trace[] == 7

end

@testset "docstrings" begin
    io = IOBuffer()
    print(io, @doc model)
    @test String(take!(io)) == "my documentation\n"
end

@testset "one-line definitions" begin

@gen (static) foo(x) = (y = @trace(normal(x, 1), :y); return y)

load_generated_functions()

tr, w = generate(foo, (0,), choicemap(:y => 1))
@test get_retval(tr) == 1
@test isapprox(w, logpdf(normal, 1, 0, 1))

end

@testset "module-defined static functions" begin

tr, w = generate(MyModuleA.foo, (0,), choicemap(:y => 1))
@test get_retval(tr) == 1
@test isapprox(w, logpdf(normal, 1, 0, 1))

tr, w = generate(MyModuleB.foo, (0,), choicemap(:y => 1))
@test get_retval(tr) == 1
@test isapprox(w, logpdf(normal, 1, 0, 1))

end

@testset "static gen function choicemaps" begin
@gen (static) function bar2()
    b ~ normal(0, 1)
    return b
end
@gen (static) function bar1()
    a ~ bar2()
    x ~ normal(0, 1)
    return x
end
Gen.load_generated_functions()
tr = simulate(bar1, ())
ch = get_choices(tr)
@test has_value(ch, :x)
@test !has_value(ch, :y)
@test_throws KeyError get_submap(ch, :x)
@test has_value(get_submap(ch, :a), :b)
@test get_submap(ch, :y) == EmptyChoiceMap()
@test length(get_values_shallow(ch)) == 1
@test length(get_submaps_shallow(ch)) == 1
end

@testset "serialization" begin
    tr = simulate(model, ([1., 2., 3., 4.],))
    @test Gen.to_serializable_trace(tr) isa Gen.GenericSerializableTrace
    io = IOBuffer()
    serialize_trace(io, tr)
    seek(io, 0)
    deserialized_tr = deserialize_trace(io, model)
    @test get_choices(deserialized_tr) == get_choices(tr)
end

end # @testset "static DSL"
