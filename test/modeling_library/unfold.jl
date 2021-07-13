@testset "unfold combinator" begin

    std = 1.0

    @gen (static) function kernel(t::Int, x_prev::Float64, (grad)(alpha::Float64), (grad)(beta::Float64))
        x = @trace(normal(x_prev * alpha + beta, std), :x)
        return x
    end

    Gen.load_generated_functions()

    foo = Unfold(kernel)

    @testset "Julia call" begin
        @test length(foo(5, 0., 1.0, 1.0)) == 5
    end

    @testset "serialization" begin
        tr = simulate(foo, (3, 0.1, 0.2, 0.3))
        @test serialize_loop_successful(tr)
    end

    @testset "simulate" begin
        x_init = 0.1
        alpha = 0.2
        beta = 0.3
        x1 = 1.1
        x2 = 1.2
        x3 = 1.3
        trace = simulate(foo, (3, x_init, alpha, beta))
        x1 = trace[1 => :x]
        x2 = trace[2 => :x]
        x3 = trace[3 => :x]
        choices = get_choices(trace)
        @test length(collect(get_values_shallow(choices))) == 0
        @test length(collect(get_submaps_shallow(choices))) == 3
        expected_score = (logpdf(normal, x1, x_init * alpha + beta, std)
                          + logpdf(normal, x2, x1 * alpha + beta, std)
                          + logpdf(normal, x3, x2 * alpha + beta, std))
        @test isapprox(get_score(trace), expected_score)
        retval = get_retval(trace)
        @test length(retval) == 3
        @test retval[1] == x1
        @test retval[2] == x2
        @test retval[3] == x3
    end

    @testset "generate" begin
        x_init = 0.1
        alpha = 0.2
        beta = 0.3
        x1 = 1.1
        x2 = 1.2
        x3 = 1.3
        constraints = choicemap()
        constraints[1 => :x] = x1
        constraints[3 => :x] = x3
        (trace, weight) = generate(foo, (3, x_init, alpha, beta), constraints)
        choices = get_choices(trace)
        @test choices[1 => :x] == x1
        @test choices[3 => :x] == x3
        @test length(collect(get_values_shallow(choices))) == 0
        @test length(collect(get_submaps_shallow(choices))) == 3
        x2 = choices[2 => :x]
        expected_weight = (logpdf(normal, x1, x_init * alpha + beta, std)
                           + logpdf(normal, x3, x2 * alpha + beta, std))
        expected_score = (logpdf(normal, x1, x_init * alpha + beta, std)
                          + logpdf(normal, x2, x1 * alpha + beta, std)
                          + logpdf(normal, x3, x2 * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        @test isapprox(get_score(trace), expected_score)
        retval = get_retval(trace)
        @test length(retval) == 3
        @test retval[1] == x1
        @test retval[2] == x2
        @test retval[3] == x3
    end

    @testset "propose" begin
        x_init = 0.1
        alpha = 0.2
        beta = 0.3
        (choices, weight, retval) = propose(foo, (3, x_init, alpha, beta))
        @test length(collect(get_values_shallow(choices))) == 0
        @test length(collect(get_submaps_shallow(choices))) == 3
        x1 = choices[1 => :x]
        x2 = choices[2 => :x]
        x3 = choices[3 => :x]
        expected_weight = (logpdf(normal, x1, x_init * alpha + beta, std)
                           + logpdf(normal, x2, x1 * alpha + beta, std)
                           + logpdf(normal, x3, x2 * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        @test length(retval) == 3
        @test retval[1] == x1
        @test retval[2] == x2
        @test retval[3] == x3
    end

    @testset "assess" begin
        x_init = 0.1
        alpha = 0.2
        beta = 0.3
        x1 = 1.1
        x2 = 1.2
        x3 = 1.3
        choices = choicemap()
        choices[1 => :x] = x1
        choices[2 => :x] = x2
        choices[3 => :x] = x3
        (weight, retval) = assess(foo, (3, x_init, alpha, beta), choices)
        expected_weight = (logpdf(normal, x1, x_init * alpha + beta, std)
                           + logpdf(normal, x2, x1 * alpha + beta, std)
                           + logpdf(normal, x3, x2 * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        @test length(retval) == 3
        @test retval[1] == x1
        @test retval[2] == x2
        @test retval[3] == x3
    end

    @testset "update" begin
        x_init = 0.1
        alpha = 0.2
        beta = 0.3
        x1 = 1.1
        x2 = 1.2

        function get_initial_trace()
            constraints = choicemap()
            constraints[1 => :x] = x1
            constraints[2 => :x] = x2
            (trace, _) = generate(foo, (2, x_init, alpha, beta), constraints)
            trace
        end

        # unknown change to args, increasing length from 2 to 3 and change 2 and change params
        trace = get_initial_trace()
        x2_new = 1.3
        x3_new = 1.4
        alpha_new = 0.5
        constraints = choicemap()
        constraints[2 => :x] = x2_new
        constraints[3 => :x] = x3_new
        (trace, weight, retdiff, discard) = update(trace,
                                                   (3, x_init, alpha_new, beta),
                                                   (UnknownChange(), UnknownChange(), UnknownChange(), UnknownChange()), constraints)
        choices = get_choices(trace)
        @test get_args(trace) == (3, x_init, alpha_new, beta)
        @test choices[1 => :x] == x1
        @test choices[2 => :x] == x2_new
        @test choices[3 => :x] == x3_new
        @test discard[2 => :x] == x2
        expected_score = (logpdf(normal, x1, x_init * alpha_new + beta, std)
                          + logpdf(normal, x2_new, x1 * alpha_new + beta, std)
                          + logpdf(normal, x3_new, x2_new * alpha_new + beta, std))
        @test isapprox(get_score(trace), expected_score)
        expected_weight = (logpdf(normal, x3_new, x2_new * alpha_new + beta, std)
                           + logpdf(normal, x2_new, x1 * alpha_new + beta, std)
                           - logpdf(normal, x2, x1 * alpha + beta, std)
                           + logpdf(normal, x1, x_init * alpha_new + beta, std)
                           - logpdf(normal, x1, x_init * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        retval = get_retval(trace)
        @test length(retval) == 3
        @test retval[1] == x1
        @test retval[2] == x2_new
        @test retval[3] == x3_new
        @test isa(retdiff, VectorDiff)
        @test !haskey(retdiff.updated, 1) # no diff
        @test retdiff.updated[2] == UnknownChange() # x changed
        @test !haskey(retdiff.updated, 3) # new, not retained

        # unknown change to args, decreasing length from 2 to 1 and change 1 and change params
        trace = get_initial_trace()
        x1_new = 1.3
        alpha_new = 0.5
        constraints = choicemap()
        constraints[1 => :x] = x1_new
        (trace, weight, retdiff, discard) = update(trace,
                                                   (1, x_init, alpha_new, beta),
                                                   (UnknownChange(), UnknownChange(), UnknownChange(), UnknownChange()), constraints)
        choices = get_choices(trace)
        @test get_args(trace) == (1, x_init, alpha_new, beta)
        @test !has_value(choices, 2 => :x)
        @test !has_value(choices, 3 => :x)
        @test choices[1 => :x] == x1_new
        @test discard[1 => :x] == x1
        @test discard[2 => :x] == x2
        @test isapprox(get_score(trace), logpdf(normal, x1_new, x_init * alpha_new + beta, std))
        expected_weight = (logpdf(normal, x1_new, x_init * alpha_new + beta, std)
                           - logpdf(normal, x1, x_init * alpha + beta, std)
                           - logpdf(normal, x2, x1 * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        retval = get_retval(trace)
        @test length(retval) == 1
        @test retval[1] == x1_new
        @test isa(retdiff, VectorDiff)
        @test retdiff.updated[1] == UnknownChange()
        @test !haskey(retdiff.updated, 2) # removed

        # increasing length from 2 to 4; constrain 4 and let 3 be generated from prior
        # also change alpha
        trace = get_initial_trace()
        x4_new = 1.3
        alpha_new = 0.5
        constraints = choicemap()
        constraints[4 => :x] = x4_new
        (trace, weight, retdiff, discard) = update(trace,
                                                   (4, x_init, alpha_new, beta),
                                                   (UnknownChange(), UnknownChange(), UnknownChange(), UnknownChange()), constraints)
        choices = get_choices(trace)
        @test isempty(discard)
        @test get_args(trace) == (4, x_init, alpha_new, beta)
        @test choices[1 => :x] == x1
        @test choices[2 => :x] == x2
        x3_new = choices[3 => :x]
        @test choices[4 => :x] == x4_new
        expected_score = (logpdf(normal, x1, x_init * alpha_new + beta, std)
                          + logpdf(normal, x2, x1 * alpha_new + beta, std)
                          + logpdf(normal, x3_new, x2 * alpha_new + beta, std)
                          + logpdf(normal, x4_new, x3_new * alpha_new + beta, std))
        @test isapprox(get_score(trace), expected_score)
        expected_weight = (logpdf(normal, x1, x_init * alpha_new + beta, std)
                           - logpdf(normal, x1, x_init * alpha + beta, std)
                           + logpdf(normal, x2, x1 * alpha_new + beta, std)
                           - logpdf(normal, x2, x1 * alpha + beta, std)
                           + logpdf(normal, x4_new, x3_new * alpha_new + beta, std))
        @test isapprox(weight, expected_weight)
        retval = get_retval(trace)
        @test length(retval) == 4
        @test retval[1] == x1
        @test retval[2] == x2
        @test retval[3] == x3_new
        @test retval[4] == x4_new
        @test isa(retdiff, VectorDiff)
        @test !haskey(retdiff.updated, 1) # no diff
        @test !haskey(retdiff.updated, 2) # no diff
        @test !haskey(retdiff.updated, 3) # new
        @test !haskey(retdiff.updated, 4) # new

        # no change to arguments, change nothing
        trace = get_initial_trace()
        (trace, weight, retdiff, discard) = update(trace,
                                                   (2, x_init, alpha, beta),
                                                   (NoChange(), NoChange(), NoChange(), NoChange()), EmptyChoiceMap())
        choices = get_choices(trace)
        @test get_args(trace) == (2, x_init, alpha, beta)
        @test choices[1 => :x] == x1
        @test choices[2 => :x] == x2
        @test isempty(discard)
        expected_score = (logpdf(normal, x1, x_init * alpha + beta, std)
                          + logpdf(normal, x2, x1 * alpha + beta, std))
        @test isapprox(get_score(trace), expected_score)
        @test isapprox(weight, 0.)
        retval = get_retval(trace)
        @test retdiff == NoChange()

        # no change to arguments, change x2
        trace = get_initial_trace()
        x2_new = 3.3
        constraints = choicemap()
        constraints[2 => :x] = x2_new
        (trace, weight, retdiff, discard) = update(trace,
                                                   (2, x_init, alpha, beta),
                                                   (NoChange(), NoChange(), NoChange(), NoChange()), constraints)
        choices = get_choices(trace)
        @test get_args(trace) == (2, x_init, alpha, beta)
        @test choices[1 => :x] == x1
        @test choices[2 => :x] == x2_new
        @test discard[2 => :x] == x2
        expected_score = (logpdf(normal, x1, x_init * alpha + beta, std)
                          + logpdf(normal, x2_new, x1 * alpha + beta, std))
        @test isapprox(get_score(trace), expected_score)
        expected_weight = (logpdf(normal, x2_new, x1 * alpha + beta, std)
                           - logpdf(normal, x2, x1 * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        retval = get_retval(trace)
        @test length(retval) == 2
        @test retval[1] == x1
        @test retval[2] == x2_new
        @test isa(retdiff, VectorDiff)
        @test !haskey(retdiff.updated, 1) # no change
        @test retdiff.updated[2] == UnknownChange()

        # init_changed=true, params_changed=false, change nothing
        trace = get_initial_trace()
        x_init_new = 0.1
        (trace, weight, retdiff, discard) = update(trace,
                                                   (2, x_init_new, alpha, beta),
                                                   (NoChange(), UnknownChange(), NoChange(), NoChange()), EmptyChoiceMap())
        choices = get_choices(trace)
        @test get_args(trace) == (2, x_init_new, alpha, beta)
        @test choices[1 => :x] == x1
        @test choices[2 => :x] == x2
        @test isempty(discard)
        expected_score = (logpdf(normal, x1, x_init_new * alpha + beta, std)
                          + logpdf(normal, x2, x1 * alpha + beta, std))
        @test isapprox(get_score(trace), expected_score)
        expected_weight = (logpdf(normal, x1, x_init_new * alpha + beta, std)
                           - logpdf(normal, x1, x_init * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        retval = get_retval(trace)
        @test length(retval) == 2
        @test retval[1] == x1
        @test retval[2] == x2
        @test retdiff == NoChange()

        # init_changed=false, params_changed=true, change nothing
        trace = get_initial_trace()
        alpha_new = 0.5
        (trace, weight, retdiff, discard) = update(trace,
                                                   (2, x_init, alpha_new, beta),
                                                   (NoChange(), NoChange(), UnknownChange(), UnknownChange()), EmptyChoiceMap())
        choices = get_choices(trace)
        @test get_args(trace) == (2, x_init, alpha_new, beta)
        @test choices[1 => :x] == x1
        @test choices[2 => :x] == x2
        @test isempty(discard)
        expected_score = (logpdf(normal, x1, x_init * alpha_new + beta, std)
                          + logpdf(normal, x2, x1 * alpha_new + beta, std))
        @test isapprox(get_score(trace), expected_score)
        expected_weight = (logpdf(normal, x1, x_init * alpha_new + beta, std)
                           - logpdf(normal, x1, x_init * alpha + beta, std)
                           + logpdf(normal, x2, x1 * alpha_new + beta, std)
                           - logpdf(normal, x2, x1 * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        retval = get_retval(trace)
        @test length(retval) == 2
        @test retval[1] == x1
        @test retval[2] == x2
        @test retdiff == NoChange()
    end

    @testset "regenerate" begin
        x_init = 0.1
        alpha = 0.2
        beta = 0.3
        x1 = 1.1
        x2 = 1.2

        function get_initial_trace()
            constraints = choicemap()
            constraints[1 => :x] = x1
            constraints[2 => :x] = x2
            (trace, _) = generate(foo, (2, x_init, alpha, beta), constraints)
            trace
        end

        # unknown change to args, increasing length from 2 to 3 and change 2 and change params
        trace = get_initial_trace()
        alpha_new = 0.5
        selection = select(2 => :x)
        (trace, weight, retdiff) = regenerate(trace,
                                              (3, x_init, alpha_new, beta),
                                              (UnknownChange(), UnknownChange(), UnknownChange(), UnknownChange()), selection)
        choices = get_choices(trace)
        @test get_args(trace) == (3, x_init, alpha_new, beta)
        @test choices[1 => :x] == x1
        x2_new = choices[2 => :x]
        x3_new = choices[3 => :x]
        expected_score = (logpdf(normal, x1, x_init * alpha_new + beta, std)
                          + logpdf(normal, x2_new, x1 * alpha_new + beta, std)
                          + logpdf(normal, x3_new, x2_new * alpha_new + beta, std))
        @test isapprox(get_score(trace), expected_score)
        expected_weight = (logpdf(normal, x1, x_init * alpha_new + beta, std)
                           - logpdf(normal, x1, x_init * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        retval = get_retval(trace)
        @test length(retval) == 3
        @test retval[1] == x1
        @test retval[2] == x2_new
        @test retval[3] == x3_new
        @test isa(retdiff, VectorDiff)
        @test !haskey(retdiff.updated, 1) # no diff
        @test retdiff.updated[2] == UnknownChange() # retval changed
        @test !haskey(retdiff.updated, 3) # new, not retained

        # unknown change to args, decreasing length from 2 to 1 and change 1 and change params
        trace = get_initial_trace()
        alpha_new = 0.5
        selection = select(1 => :x)
        (trace, weight, retdiff) = regenerate(trace,
                                              (1, x_init, alpha_new, beta),
                                              (UnknownChange(), UnknownChange(), UnknownChange(), UnknownChange()), selection)
        choices = get_choices(trace)
        @test get_args(trace) == (1, x_init, alpha_new, beta)
        @test !has_value(choices, 2 => :x)
        @test !has_value(choices, 3 => :x)
        x1_new = choices[1 => :x]
        @test isapprox(get_score(trace), logpdf(normal, x1_new, x_init * alpha_new + beta, std))
        @test isapprox(weight, 0.)
        retval = get_retval(trace)
        @test length(retval) == 1
        @test retval[1] == x1_new
        @test isa(retdiff, VectorDiff)
        @test retdiff.updated[1] == UnknownChange() # retval changed
        @test !haskey(retdiff.updated, 2) # removed

        # no change to args, change nothing
        trace = get_initial_trace()
        (trace, weight, retdiff) = regenerate(trace,
                                              (2, x_init, alpha, beta),
                                              (NoChange(), NoChange(), NoChange(), NoChange()), EmptySelection())
        choices = get_choices(trace)
        @test get_args(trace) == (2, x_init, alpha, beta)
        @test choices[1 => :x] == x1
        @test choices[2 => :x] == x2
        expected_score = (logpdf(normal, x1, x_init * alpha + beta, std)
                          + logpdf(normal, x2, x1 * alpha + beta, std))
        @test isapprox(get_score(trace), expected_score)
        @test isapprox(weight, 0.)
        retval = get_retval(trace)
        @test length(retval) == 2
        @test retval[1] == x1
        @test retval[2] == x2
        @test retdiff == NoChange()

        # no change to args, change x2
        trace = get_initial_trace()
        x2_new = 3.3
        selection = select(2 => :x)
        (trace, weight, retdiff) = regenerate(trace,
                                              (2, x_init, alpha, beta),
                                              (NoChange(), NoChange(), NoChange(), NoChange()), selection)
        choices = get_choices(trace)
        @test get_args(trace) == (2, x_init, alpha, beta)
        @test choices[1 => :x] == x1
        x2_new = choices[2 => :x]
        expected_score = (logpdf(normal, x1, x_init * alpha + beta, std)
                          + logpdf(normal, x2_new, x1 * alpha + beta, std))
        @test isapprox(get_score(trace), expected_score)
        @test isapprox(weight, 0.)
        retval = get_retval(trace)
        @test length(retval) == 2
        @test retval[1] == x1
        @test retval[2] == x2_new
        @test isa(retdiff, VectorDiff)
        @test !haskey(retdiff.updated, 1) # no diff
        @test retdiff.updated[2] == UnknownChange() # retval changed

        # init_changed=true, params_changed=false, change nothing
        trace = get_initial_trace()
        x_init_new = -0.1
        (trace, weight, retdiff) = regenerate(trace,
                                              (2, x_init_new, alpha, beta),
                                              (NoChange(), UnknownChange(), NoChange(), NoChange()), EmptySelection())
        choices = get_choices(trace)
        @test get_args(trace) == (2, x_init_new, alpha, beta)
        @test choices[1 => :x] == x1
        @test choices[2 => :x] == x2
        expected_score = (logpdf(normal, x1, x_init_new * alpha + beta, std)
                          + logpdf(normal, x2, x1 * alpha + beta, std))
        @test isapprox(get_score(trace), expected_score)
        expected_weight = (logpdf(normal, x1, x_init_new * alpha + beta, std)
                           - logpdf(normal, x1, x_init * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        retval = get_retval(trace)
        @test length(retval) == 2
        @test retval[1] == x1
        @test retval[2] == x2
        @test retdiff == NoChange()

        # init_changed=false, params_changed=true, change nothing
        trace = get_initial_trace()
        alpha_new = 0.5
        (trace, weight, retdiff) = regenerate(trace,
                                              (2, x_init, alpha_new, beta),
                                              (NoChange(), NoChange(), UnknownChange(), UnknownChange()), EmptySelection())
        choices = get_choices(trace)
        @test get_args(trace) == (2, x_init, alpha_new, beta)
        @test choices[1 => :x] == x1
        @test choices[2 => :x] == x2
        expected_score = (logpdf(normal, x1, x_init * alpha_new + beta, std)
                          + logpdf(normal, x2, x1 * alpha_new + beta, std))
        @test isapprox(get_score(trace), expected_score)
        expected_weight = (logpdf(normal, x1, x_init * alpha_new + beta, std)
                           - logpdf(normal, x1, x_init * alpha + beta, std)
                           + logpdf(normal, x2, x1 * alpha_new + beta, std)
                           - logpdf(normal, x2, x1 * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        retval = get_retval(trace)
        @test length(retval) == 2
        @test retval[1] == x1
        @test retval[2] == x2
        @test retdiff == NoChange()
    end

    @testset "test 1 - accumulate_param_gradients!" begin

        @gen function kernel(t::Int, x_prev::Float64, (grad)(alpha::Float64), (grad)(beta::Float64))
            @param std::Float64
            x = @trace(normal(x_prev * alpha + beta, std), :x)
            return x
        end

        foo = Unfold(kernel)

        std = 1.
        set_param!(kernel, :std, std)

        x_init = 0.1
        alpha = 0.2
        beta = 0.3
        x1 = 1.1
        x2 = 1.2

        constraints = choicemap()
        constraints[1 => :x] = x1
        constraints[2 => :x] = x2
        (trace, _) = generate(foo, (2, x_init, alpha, beta), constraints)

        zero_param_grad!(kernel, :std)
        input_grads = accumulate_param_gradients!(trace, nothing)
        @test input_grads[1] == nothing # length
        @test input_grads[2] == nothing # inital state
        #@test isapprox(input_grads[3], expected_xs_grad) # alpha
        #@test isapprox(input_grads[4], expected_ys_grad) # beta
        expected_std_grad = (logpdf_grad(normal, x1, x_init * alpha + beta, std)[3]
                             + logpdf_grad(normal, x2, x1 * alpha + beta, std)[3])
        @test isapprox(get_param_grad(kernel, :std), expected_std_grad)
    end

    @gen (grad) function ker(t, (grad)(x::Float64))
        return 2.0 * x
    end

    @testset "test 2 - accumulate_param_gradients!" begin
        uf = Unfold(ker)
        tr = simulate(uf, (10, 1.0))
        k_grads = accumulate_param_gradients!(tr, [i == 10 ? 1.0 : nothing for i in 1 : 10])
        @test isapprox(k_grads[2], 1024.0)
    end

    @gen (grad) function kernel(t::Int, 
            (grad)(x_prev::Float64), 
            (grad)(alpha::Float64), 
            (grad)(beta::Float64))
        x = @trace(normal(x_prev * alpha + beta, 1.0), :x)
        return x
    end

    @testset "choice gradients" begin
        uf = Unfold(kernel)
        alpha, beta = 1.0, 0.0
        tr = simulate(uf, (10, 0.0, alpha, beta))
        chm = get_choices(tr)

        # Test propagation over state.
        for i in 1 : 10
            sel = select(i => :x)
            grads, value_choices, grad_choices = choice_gradients(tr, sel, [nothing for _ in 1 : 10])

            # Explicit edge cases.
            grad_t = logpdf_grad(normal, chm[i => :x], i == 1 ? 0.0 : chm[(i - 1) => :x], 1.0) # if first index, init_state = 0.0
            i == 10 ? grad_tp1 = (0.0, 0.0, 0.0) : grad_tp1 = logpdf_grad(normal, chm[(i + 1) => :x], chm[i => :x], 1.0) # if last index, no more iterations.

            # Propagation should be previous choice value grad + next choice mean grad.
            @test isapprox(grad_t[1] + grad_tp1[2], grad_choices[i => :x])
        end

        # Test propagation with non-nothing ret grad.
        for i in 1 : 10
            sel = select(i => :x)
            grads, value_choices, grad_choices = choice_gradients(tr, sel, [k == i ? 1.0 : nothing for k in 1 : 10])
            grad_t = logpdf_grad(normal, chm[i => :x], i == 1 ? 0.0 : chm[(i - 1) => :x], 1.0)
            i == 10 ? grad_tp1 = (0.0, 0.0, 0.0) : grad_tp1 = logpdf_grad(normal, chm[(i + 1) => :x], chm[i => :x], 1.0)

            # Propagation should be same as above, but including accumulating the retval grad.
            @test isapprox(grad_t[1] + grad_tp1[2] + 1.0, grad_choices[i => :x])
        end
    end
end
