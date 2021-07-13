@testset "call_at combinator" begin

    @gen (grad) function foo((grad)(x::Float64))
        return x + @trace(normal(x, 1), :y)
    end

    at = call_at(foo, Int)

    @testset "assess" begin
        y = 1.234
        choices = choicemap()
        choices[3 => :y] = y
        (weight, value) = assess(at, (0.4, 3), choices)
        @test isapprox(weight, logpdf(normal, y, 0.4, 1))
        @test isapprox(value, 0.4 + y)
    end

    @testset "propose" begin
        (choices, weight, value) = propose(at, (0.4, 3))
        y = choices[3 => :y]
        @test isapprox(weight, logpdf(normal, y, 0.4, 1))
        @test length(collect(get_values_shallow(choices))) == 0
        @test length(collect(get_submaps_shallow(choices))) == 1
    end

    @testset "generate" begin

        # without constraint
        (trace, weight) = generate(at, (0.4, 3), EmptyChoiceMap())
        choices = get_choices(trace)
        @test weight == 0.
        y = choices[3 => :y]
        @test get_retval(trace) == 0.4 + y
        @test length(collect(get_values_shallow(choices))) == 0
        @test length(collect(get_submaps_shallow(choices))) == 1

        # with constraints
        y = 1.234
        constraints = choicemap()
        constraints[3 => :y] = y
        (trace, weight) = generate(at, (0.4, 3), constraints)
        choices = get_choices(trace)
        @test choices[3 => :y] == y
        @test get_retval(trace) == 0.4 + y
        @test isapprox(weight, logpdf(normal, y, 0.4, 1.))
        @test length(collect(get_values_shallow(choices))) == 0
        @test length(collect(get_submaps_shallow(choices))) == 1
    end

    function get_trace()
        y = 1.234
        constraints = choicemap()
        constraints[3 => :y] = y
        (trace, _) = generate(at, (0.4, 3), constraints)
        (trace, y)
    end

    @testset "serialization" begin
        @test serialize_loop_successful(get_trace()[1])    
    end

    @testset "project" begin
        (trace, y) = get_trace()
        @test isapprox(project(trace, EmptySelection()), 0.)
        selection = select(3 => :y)
        @test isapprox(project(trace, selection), logpdf(normal, y, 0.4, 1))
    end

    @testset "force update" begin
        (trace, y) = get_trace()

        # change kernel_args, same key, no constraint
        (new_trace, weight, retdiff, discard) = update(trace,
            (0.2, 3), (UnknownChange(), UnknownChange()), EmptyChoiceMap())
        choices = get_choices(new_trace)
        @test choices[3 => :y] == y
        @test length(collect(get_values_shallow(choices))) == 0
        @test length(collect(get_submaps_shallow(choices))) == 1
        @test isapprox(weight, logpdf(normal, y, 0.2, 1) - logpdf(normal, y, 0.4, 1))
        @test get_retval(new_trace) == 0.2 + y
        @test isempty(discard)
        @test isapprox(get_score(new_trace), logpdf(normal, y, 0.2, 1))

        # change kernel_args, same key, with constraint
        constraints = choicemap()
        y_new = 2.345
        constraints[3 => :y] = y_new
        (new_trace, weight, retdiff, discard) = update(trace,
            (0.2, 3), (UnknownChange(), UnknownChange()), constraints)
        choices = get_choices(new_trace)
        @test choices[3 => :y] == y_new
        @test length(collect(get_values_shallow(choices))) == 0
        @test length(collect(get_submaps_shallow(choices))) == 1
        @test isapprox(weight, logpdf(normal, y_new, 0.2, 1) - logpdf(normal, y, 0.4, 1))
        @test get_retval(new_trace) == 0.2 + y_new
        @test discard[3 => :y] == y
        @test length(collect(get_values_shallow(discard))) == 0
        @test length(collect(get_submaps_shallow(discard))) == 1
        @test isapprox(get_score(new_trace), logpdf(normal, y_new, 0.2, 1))

        # change kernel_args, different key, with constraint
        y_new = 2.345
        constraints = choicemap()
        constraints[4 => :y] = y_new
        (new_trace, weight, retdiff, discard) = update(trace,
            (0.2, 4), (UnknownChange(), UnknownChange()), constraints)
        choices = get_choices(new_trace)
        @test choices[4 => :y] == y_new
        @test length(collect(get_values_shallow(choices))) == 0
        @test length(collect(get_submaps_shallow(choices))) == 1
        @test isapprox(weight, logpdf(normal, y_new, 0.2, 1) - logpdf(normal, y, 0.4, 1))
        @test get_retval(new_trace) == 0.2 + y_new
        @test discard[3 => :y] == y
        @test length(collect(get_values_shallow(discard))) == 0
        @test length(collect(get_submaps_shallow(discard))) == 1
        @test isapprox(get_score(new_trace), logpdf(normal, y_new, 0.2, 1))
    end

    @testset "free update" begin
        (trace, y) = get_trace()

        # change kernel_args, same key, not selected
        (new_trace, weight, retdiff) = regenerate(trace,
            (0.2, 3), (UnknownChange(), UnknownChange()), EmptySelection())
        choices = get_choices(new_trace)
        @test choices[3 => :y] == y
        @test length(collect(get_values_shallow(choices))) == 0
        @test length(collect(get_submaps_shallow(choices))) == 1
        @test isapprox(weight, logpdf(normal, y, 0.2, 1) - logpdf(normal, y, 0.4, 1))
        @test get_retval(new_trace) == 0.2 + y
        @test isapprox(get_score(new_trace), logpdf(normal, y, 0.2, 1))

        # change kernel_args, same key, selected
        selection = select(3 => :y)
        (new_trace, weight, retdiff) = regenerate(trace,
            (0.2, 3), (UnknownChange(), UnknownChange()), selection)
        choices = get_choices(new_trace)
        y_new = choices[3 => :y]
        @test length(collect(get_values_shallow(choices))) == 0
        @test length(collect(get_submaps_shallow(choices))) == 1
        @test weight == 0.
        @test get_retval(new_trace) == 0.2 + y_new
        @test isapprox(get_score(new_trace), logpdf(normal, y_new, 0.2, 1))

        # change kernel_args, different key, not selected
        (new_trace, weight, retdiff) = regenerate(trace,
            (0.2, 4), (UnknownChange(), UnknownChange()), EmptySelection())
        choices = get_choices(new_trace)
        y_new = choices[4 => :y]
        @test length(collect(get_values_shallow(choices))) == 0
        @test length(collect(get_submaps_shallow(choices))) == 1
        @test weight == 0.
        @test get_retval(new_trace) == 0.2 + y_new
        @test isapprox(get_score(new_trace), logpdf(normal, y_new, 0.2, 1))
    end

    @testset "choice_gradients" begin
        (trace, y) = get_trace()

        # not selected
        retval_grad = 1.234
        (input_grads, choices, gradients) = choice_gradients(
            trace, EmptySelection(), retval_grad)
        @test isempty(choices)
        @test isempty(gradients)
        @test length(input_grads) == 2
        @test isapprox(input_grads[1], logpdf_grad(normal, y, 0.4, 1.0)[2] + retval_grad)
        @test input_grads[2] == nothing # the key has no gradient

        # selected with retval_grad
        retval_grad = 1.234
        selection = select(3 => :y)
        (input_grads, choices, gradients) = choice_gradients(
            trace, selection, retval_grad)
        @test choices[3 => :y] == y
        @test isapprox(gradients[3 => :y], logpdf_grad(normal, y, 0.4, 1.0)[1] + retval_grad)
        @test length(collect(get_values_shallow(gradients))) == 0
        @test length(collect(get_submaps_shallow(gradients))) == 1
        @test length(collect(get_values_shallow(choices))) == 0
        @test length(collect(get_submaps_shallow(choices))) == 1
        @test length(input_grads) == 2
        @test isapprox(input_grads[1], logpdf_grad(normal, y, 0.4, 1.0)[2] + retval_grad)
        @test input_grads[2] == nothing # the key has no gradient
    end

    @testset "accumulate_param_gradients!" begin
        (trace, y) = get_trace()
        retval_grad = 1.234
        input_grads = accumulate_param_gradients!(trace, retval_grad)
        @test length(input_grads) == 2
        @test isapprox(input_grads[1], logpdf_grad(normal, y, 0.4, 1)[2] + retval_grad)
        @test input_grads[2] == nothing # the key has no gradient
    end
end
