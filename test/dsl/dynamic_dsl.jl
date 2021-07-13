using Gen: AddressVisitor, all_visited, visit!, get_visited

struct DummyReturnType
end

@testset "Dynamic DSL" begin

@testset "Julia call syntax" begin

    @gen function foo()
        return 1
    end

    @test foo() == 1

    @gen function bar(x=1)
        return x
    end

    @test bar() == 1

    @gen function baz(x::Int=1)
        return x
    end

    @test baz(5) == 5

    @gen (grad) oneliner(x::Float64, (grad)(y::Float64=5.0)) = x+y

    @test oneliner(5.0) == 10.0

    @gen outer() = @trace(inner())
    @gen inner() = 5.0

    @test outer() == 5.0
end

@testset "return type" begin

    @gen function foo()::DummyReturnType
        return DummyReturnType()
    end

    @test Gen.get_return_type(foo) == DummyReturnType
end

# TODO also test get_unvisited

@testset "address visitor" begin
    choices = choicemap()
    choices[:x] = 1
    choices[:y => :z] = 2

    visitor = AddressVisitor()
    visit!(visitor, :x)
    visit!(visitor, :y => :z)
    @test all_visited(get_visited(visitor), choices)

    visitor = AddressVisitor()
    visit!(visitor, :x)
    @test !all_visited(get_visited(visitor), choices)

    visitor = AddressVisitor()
    visit!(visitor, :y => :z)
    @test !all_visited(get_visited(visitor), choices)

    visitor = AddressVisitor()
    visit!(visitor, :x)
    visit!(visitor, :y)
    @test all_visited(get_visited(visitor), choices)
end

@testset "simulate" begin

    @gen function foo(p)
        return @trace(bernoulli(p), :x)
    end

    p = 0.4
    trace = simulate(foo, (p,))
    @test trace[:x] == get_retval(trace)
    @test get_args(trace) == (p,)
    @test get_score(trace) == (trace[:x] ? log(p) : log(1 - p))
end

@testset "update" begin
    @gen function bar()
        @trace(normal(0, 1), :a)
    end

    @gen function baz()
        @trace(normal(0, 1), :b)
    end

    @gen function foo()
        if @trace(bernoulli(0.4), :branch)
            @trace(normal(0, 1), :x)
            @trace(bar(), :u)
        else
            @trace(normal(0, 1), :y)
            @trace(baz(), :v)
        end
    end

    # get a trace which follows the first branch
    constraints = choicemap()
    constraints[:branch] = true
    (trace,) = generate(foo, (), constraints)
    x = get_choices(trace)[:x]
    a = get_choices(trace)[:u => :a]

    # force to follow the second branch
    y = 1.123
    b = -2.1
    constraints = choicemap()
    constraints[:branch] = false
    constraints[:y] = y
    constraints[:v => :b] = b
    (new_trace, weight, retdiff, discard) = update(trace,
        (), (), constraints)

    # test discard
    @test get_value(discard, :branch) == true
    @test get_value(discard, :x) == x
    @test get_value(discard, :u => :a) == a
    @test length(collect(get_values_shallow(discard))) == 2
    @test length(collect(get_submaps_shallow(discard))) == 1

    # test new trace
    new_assignment = get_choices(new_trace)
    @test get_value(new_assignment, :branch) == false
    @test get_value(new_assignment, :y) == y
    @test get_value(new_assignment, :v => :b) == b
    @test length(collect(get_values_shallow(new_assignment))) == 2
    @test length(collect(get_submaps_shallow(new_assignment))) == 1

    # test score and weight
    prev_score = (
        logpdf(bernoulli, true, 0.4) +
        logpdf(normal, x, 0, 1) +
        logpdf(normal, a, 0, 1))
    expected_new_score = (
        logpdf(bernoulli, false, 0.4) +
        logpdf(normal, y, 0, 1) +
        logpdf(normal, b, 0, 1))
    expected_weight = expected_new_score - prev_score
    @test isapprox(expected_new_score, get_score(new_trace))
    @test isapprox(expected_weight, weight)

    # test retdiff
    @test retdiff === UnknownChange()

    # Addresses under the :data namespace will be visited,
    # but nothing there will be discarded.
    @gen function loopy()
        a = @trace(normal(0, 1), :a)
        for i=1:5
            @trace(normal(a, 1), :data => i)
        end
    end

    # Get an initial trace
    constraints = choicemap()
    constraints[:a] = 0
    for i=1:5
        constraints[:data => i] = 0
    end
    (trace,) = generate(loopy, (), constraints)

    # Update a
    constraints = choicemap()
    constraints[:a] = 1
    (new_trace, weight, retdiff, discard) = update(trace,
        (), (), constraints)

    # Test discard, score, weight, retdiff
    @test get_value(discard, :a) == 0
    prev_score = logpdf(normal, 0, 0, 1) * 6
    expected_new_score = logpdf(normal, 1, 0, 1) + 5 * logpdf(normal, 0, 1, 1)
    expected_weight = expected_new_score - prev_score
    @test isapprox(expected_new_score, get_score(new_trace))
    @test isapprox(expected_weight, weight)
    @test retdiff === UnknownChange()
end

@testset "regenerate" begin
    @gen function bar(mu)
        @trace(normal(mu, 1), :a)
    end

    @gen function baz(mu)
        @trace(normal(mu, 1), :b)
    end

    @gen function foo(mu)
        if @trace(bernoulli(0.4), :branch)
            @trace(normal(mu, 1), :x)
            @trace(bar(mu), :u)
        else
            @trace(normal(mu, 1), :y)
            @trace(baz(mu), :v)
        end
    end

    # get a trace which follows the first branch
    mu = 0.123
    constraints = choicemap()
    constraints[:branch] = true
    (trace,) = generate(foo, (mu,), constraints)
    x = get_choices(trace)[:x]
    a = get_choices(trace)[:u => :a]

    # resimulate branch
    selection = select(:branch)

    # try 10 times, so we are likely to get both a stay and a switch
    for i=1:10
        prev_assignment = get_choices(trace)

        # change the argument so that the weights can be nonzer
        prev_mu = mu
        mu = rand()
        (trace, weight, retdiff) = regenerate(trace,
            (mu,), (UnknownChange(),), selection)
        assignment = get_choices(trace)

        # test score
        if assignment[:branch]
            expected_score = (
                logpdf(normal, assignment[:x], mu, 1)
                + logpdf(normal, assignment[:u => :a], mu, 1)
                + logpdf(bernoulli, true, 0.4))
        else
            expected_score = (
                logpdf(normal, assignment[:y], mu, 1)
                + logpdf(normal, assignment[:v => :b], mu, 1)
                + logpdf(bernoulli, false, 0.4))
        end
        @test isapprox(expected_score, get_score(trace))

        # test values
        if assignment[:branch]
            @test has_value(assignment, :x)
            @test !isempty(get_submap(assignment, :u))
        else
            @test has_value(assignment, :y)
            @test !isempty(get_submap(assignment, :v))
        end
        @test length(collect(get_values_shallow(assignment))) == 2
        @test length(collect(get_submaps_shallow(assignment))) == 1

        # test weight
        if assignment[:branch] == prev_assignment[:branch]
            if assignment[:branch]
                expected_weight = (
                    logpdf(normal, assignment[:x], mu, 1)
                    + logpdf(normal, assignment[:u => :a], mu, 1))
                expected_weight -= (
                    logpdf(normal, assignment[:x], prev_mu, 1)
                    + logpdf(normal, assignment[:u => :a], prev_mu, 1))
            else
                expected_weight = (
                    logpdf(normal, assignment[:y], mu, 1)
                    + logpdf(normal, assignment[:v => :b], mu, 1))
                expected_weight -= (
                    logpdf(normal, assignment[:y], prev_mu, 1)
                    + logpdf(normal, assignment[:v => :b], prev_mu, 1))
            end
        else
            expected_weight = 0.
        end
        @test isapprox(expected_weight, weight)

        # test retdiff
        @test retdiff === UnknownChange()
    end

end

@testset "choice_gradients and accumulate_param_gradients!" begin

    @gen (grad) function bar((grad)(mu_z::Float64))
        @param theta1::Float64
        local z
        z = @trace(normal(mu_z + theta1, 1), :z)
        return z + mu_z
    end

    @gen (grad) function foo((grad)(mu_a::Float64))
        @param theta2::Float64
        local a, b, c
        a = @trace(normal(mu_a, 1), :a)
        b = @trace(normal(a, 1), :b)
        c = a * b * @trace(bar(a), :bar)
        return @trace(normal(c, 1), :out) + (theta2 * 3)
    end

    init_param!(bar, :theta1, 0.)
    init_param!(foo, :theta2, 0.)

    function f(mu_a, a, b, z, out)
        lpdf = 0.
        mu_z = a
        lpdf += logpdf(normal, z, mu_z, 1)
        lpdf += logpdf(normal, a, mu_a, 1)
        lpdf += logpdf(normal, b, a, 1)
        c = a * b * (z + mu_z)
        lpdf += logpdf(normal, out, c, 1)
        return lpdf + 2 * out
    end

    mu_a = 1.
    a = 2.
    b = 3.
    z = 4.
    out = 5.

    # get the initial trace
    constraints = choicemap()
    constraints[:a] = a
    constraints[:b] = b
    constraints[:out] = out
    constraints[:bar => :z] = z
    (trace, _) = generate(foo, (mu_a,), constraints)

    # compute gradients using choice_gradients
    selection = select(:bar => :z, :a, :out)
    retval_grad = 2.
    ((mu_a_grad,), choices, gradients) = choice_gradients(
        trace, selection, retval_grad)

    # check input gradient
    @test isapprox(mu_a_grad, finite_diff(f, (mu_a, a, b, z, out), 1, dx))

    # check value trie
    @test get_value(choices, :a) == a
    @test get_value(choices, :out) == out
    @test get_value(choices, :bar => :z) == z
    @test !has_value(choices, :b) # was not selected
    @test length(collect(get_submaps_shallow(choices))) == 1
    @test length(collect(get_values_shallow(choices))) == 2

    # check gradient trie
    @test length(collect(get_submaps_shallow(gradients))) == 1
    @test length(collect(get_values_shallow(gradients))) == 2
    @test !has_value(gradients, :b) # was not selected
    @test isapprox(get_value(gradients, :bar => :z),
        finite_diff(f, (mu_a, a, b, z, out), 4, dx))
    @test isapprox(get_value(gradients, :out),
        finite_diff(f, (mu_a, a, b, z, out), 5, dx))

    # compute gradients using accumulate_param_gradients!
    selection = select(:bar => :z, :a, :out)
    retval_grad = 2.
    (mu_a_grad,) = accumulate_param_gradients!(trace, retval_grad)

    # check input gradient
    @test isapprox(mu_a_grad, finite_diff(f, (mu_a, a, b, z, out), 1, dx))

    # check parameter gradient
    theta1_grad = get_param_grad(bar, :theta1)
    theta2_grad = get_param_grad(foo, :theta2)
    @test isapprox(theta1_grad, logpdf_grad(normal, z, a, 1)[2])
    @test isapprox(theta2_grad, 3 * 2)

end

@testset "backprop params with splice" begin

    @gen (grad) function baz()
        @param theta::Float64
        return theta
    end

    init_param!(baz, :theta, 0.)

    @gen (grad) function foo()
        return @trace(baz())
    end

    (trace, _) = generate(foo, ())
    retval_grad = 2.
    accumulate_param_gradients!(trace, retval_grad)
    @test isapprox(get_param_grad(baz, :theta), retval_grad)
end

@testset "gradient descent with fixed step size" begin
    @gen (grad) function foo()
        @param theta::Float64
        return theta
    end
    init_param!(foo, :theta, 0.)
    (trace, ) = generate(foo, ())
    accumulate_param_gradients!(trace, 1.)
    conf = FixedStepGradientDescent(0.001)
    state = Gen.init_update_state(conf, foo, [:theta])
    Gen.apply_update!(state)
    @test isapprox(get_param(foo, :theta), 0.001)
    @test isapprox(get_param_grad(foo, :theta), 0.)
end

@testset "gradient descent with shrinking step size" begin
    @gen (grad) function foo()
        @param theta::Float64
        return theta
    end
    init_param!(foo, :theta, 0.)
    (trace, ) = generate(foo, ())
    accumulate_param_gradients!(trace, 1.)
    conf = GradientDescent(0.001, 1000)
    state = Gen.init_update_state(conf, foo, [:theta])
    Gen.apply_update!(state)
    @test isapprox(get_param(foo, :theta), 0.001)
    @test isapprox(get_param_grad(foo, :theta), 0.)
end

@testset "multi-component addresses" begin
    @gen function bar()
        @trace(normal(0, 1), :z)
    end

    @gen function foo()
        @trace(normal(0, 1), :y)
        @trace(normal(0, 1), :x => 1)
        @trace(normal(0, 1), :x => 2)
        @trace(bar(), :x => 3)
    end

    trace, _ =  generate(foo, (), choicemap((:x => 1, 1), (:x => 2, 2), (:x => 3 => :z, 3)))
    @test trace[:x => 1] == 1
    @test trace[:x => 2] == 2
    @test trace[:x => 3 => :z] == 3

    choices = get_choices(trace)
    @test choices[:x => 1] == 1
    @test choices[:x => 2] == 2
    @test choices[:x => 3 => :z] == 3
    @test length(collect(get_values_shallow(choices))) == 1 # :y
    @test length(collect(get_submaps_shallow(choices))) == 1 # :x

    submap = get_submap(choices, :x)
    @test submap[1] == 1
    @test submap[2] == 2
    @test submap[3 => :z] == 3
    @test length(collect(get_values_shallow(submap))) == 2 # 1, 2
    @test length(collect(get_submaps_shallow(submap))) == 1 # 3

    bar_submap = get_submap(submap, 3)
    @test bar_submap[:z] == 3
end

@testset "project" begin
    @gen function bar()
        @trace(normal(0, 1), :x)
    end

    @gen function foo()
        @trace(normal(0, 2), :y)
        @trace(bar(), :z)
    end

    tr = simulate(foo, ())

    x = tr[:z => :x]
    y = tr[:y]

    @test isapprox(project(tr, select()), 0.)
    @test isapprox(project(tr, select(:y)), logpdf(normal, y, 0, 2))
    @test isapprox(project(tr, select(:z => :x)), logpdf(normal, x, 0, 1))
    @test isapprox(project(tr, select(:z => :x, :y)),
            logpdf(normal, x, 0, 1) + logpdf(normal, y, 0, 2))
end

@testset "getindex(trace)" begin
    @gen function bar(r)
        @trace(normal(0, 1), :a)
        return r
    end

    @gen function foo()
        @trace(bar(1), :x)
        @trace(bar(2), :y => :z)
        @trace(normal(0, 1), :u)
        @trace(normal(0, 1), :v => :w)
        7
    end

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

    # address that does not exist
    function test_addr_dne(addr)
        threw = false
        try
            x = trace[addr]
        catch ex
            threw = true
        end
        @test threw
    end
    test_addr_dne(:absent)
    test_addr_dne(:absent => :x)
end

@testset "docstrings" begin

    """
    my documentation
    """
    @gen function foo(x)
            return x + 1
        end

    io = IOBuffer()
    print(io, @doc foo)
    @test String(take!(io)) == "my documentation\n"

end

@testset "serialization" begin
    @gen function bar()
        @trace(normal(0, 1), :a)
    end

    @gen function baz()
        @trace(normal(0, 1), :b)
    end

    @gen function foo()
        if @trace(bernoulli(0.4), :branch)
            @trace(normal(0, 1), :x)
            @trace(bar(), :u)
        else
            @trace(normal(0, 1), :y)
            @trace(baz(), :v)
        end
    end
    tr = simulate(foo, ())
    @test serialize_loop_successful(tr)
end

end
