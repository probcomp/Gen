@testset "choice_at combinator" begin

    at = choice_at(bernoulli, Int)

    @testset "assess" begin
        choices = choicemap()
        choices[3] = true
        (weight, value) = assess(at, (0.4, 3), choices)
        @test isapprox(weight, log(0.4))
        @test value == true
    end

    @testset "propose" begin
        (choices, weight, value) = propose(at, (0.4, 3))
        @test isapprox(weight, value ? log(0.4) : log(0.6))
        @test choices[3] == value
        @test length(collect(get_values_shallow(choices))) == 1
        @test length(collect(get_submaps_shallow(choices))) == 0
    end

    @testset "generate" begin

        # without assignment
        (trace, weight) = generate(at, (0.4, 3), EmptyChoiceMap())
        @test weight == 0.
        value = get_retval(trace)
        choices = get_choices(trace)
        @test choices[3] == value
        @test length(collect(get_values_shallow(choices))) == 1
        @test length(collect(get_submaps_shallow(choices))) == 0

        # with constraints
        constraints = choicemap()
        constraints[3] = true
        (trace, weight) = generate(at, (0.4, 3), constraints)
        value = get_retval(trace)
        @test value == true
        @test isapprox(weight, log(0.4))
        choices = get_choices(trace)
        @test choices[3] == value
        @test length(collect(get_values_shallow(choices))) == 1
        @test length(collect(get_submaps_shallow(choices))) == 0
    end

    function get_trace()
        constraints = choicemap()
        constraints[3] = true
        (trace, _) = generate(at, (0.4, 3), constraints)
        trace
    end

    @testset "serialization" begin
        @test serialize_loop_successful(get_trace())
    end

    @testset "project" begin
        trace = get_trace()
        @test isapprox(project(trace, EmptySelection()), 0.)
        selection = select(3)
        @test isapprox(project(trace, selection), log(0.4))
    end

    @testset "update" begin
        trace = get_trace()

        # change kernel_args, same key, no constraint
        (new_trace, weight, retdiff, discard) = update(trace,
            (0.2, 3), (UnknownChange(), UnknownChange()), EmptyChoiceMap())
        choices = get_choices(new_trace)
        @test choices[3] == true
        @test length(collect(get_values_shallow(choices))) == 1
        @test length(collect(get_submaps_shallow(choices))) == 0
        @test isapprox(weight, log(0.2) - log(0.4))
        @test get_retval(new_trace) == true
        @test isempty(discard)

        # change kernel_args, same key, with constraint
        constraints = choicemap()
        constraints[3] = false
        (new_trace, weight, retdiff, discard) = update(trace,
            (0.2, 3), (UnknownChange(), UnknownChange()), constraints)
        choices = get_choices(new_trace)
        @test choices[3] == false
        @test length(collect(get_values_shallow(choices))) == 1
        @test length(collect(get_submaps_shallow(choices))) == 0
        @test isapprox(weight, log(1 - 0.2) - log(0.4))
        @test get_retval(new_trace) == false
        @test discard[3] == true
        @test length(collect(get_values_shallow(discard))) == 1
        @test length(collect(get_submaps_shallow(discard))) == 0

        # change kernel_args, different key, with constraint
        constraints = choicemap()
        constraints[4] = false
        (new_trace, weight, retdiff, discard) = update(trace,
            (0.2, 4), (UnknownChange(), UnknownChange()), constraints)
        choices = get_choices(new_trace)
        @test choices[4] == false
        @test length(collect(get_values_shallow(choices))) == 1
        @test length(collect(get_submaps_shallow(choices))) == 0
        @test isapprox(weight, log(1 - 0.2) - log(0.4))
        @test get_retval(new_trace) == false
        @test discard[3] == true
        @test length(collect(get_values_shallow(discard))) == 1
        @test length(collect(get_submaps_shallow(discard))) == 0
    end

    @testset "regenerate" begin
        trace = get_trace()

        # change kernel_args, same key, not selected
        (new_trace, weight, retdiff) = regenerate(trace,
            (0.2, 3), (UnknownChange(), UnknownChange()), EmptySelection())
        choices = get_choices(new_trace)
        @test choices[3] == true
        @test length(collect(get_values_shallow(choices))) == 1
        @test length(collect(get_submaps_shallow(choices))) == 0
        @test isapprox(weight, log(0.2) - log(0.4))
        @test get_retval(new_trace) == true
        @test isapprox(get_score(new_trace), log(0.2))

        # change kernel_args, same key, selected
        selection = select(3)
        (new_trace, weight, retdiff) = regenerate(trace,
            (0.2, 3), (UnknownChange(), UnknownChange()), selection)
        choices = get_choices(new_trace)
        value = choices[3]
        @test length(collect(get_values_shallow(choices))) == 1
        @test length(collect(get_submaps_shallow(choices))) == 0
        @test weight == 0.
        @test get_retval(new_trace) == value
        @test isapprox(get_score(new_trace), log(value ? 0.2 : 1 - 0.2))

        # change kernel_args, different key, not selected
        (new_trace, weight, retdiff) = regenerate(trace,
            (0.2, 4), (UnknownChange(), UnknownChange()), EmptySelection())
        choices = get_choices(new_trace)
        value = choices[4]
        @test length(collect(get_values_shallow(choices))) == 1
        @test length(collect(get_submaps_shallow(choices))) == 0
        @test weight == 0.
        @test get_retval(new_trace) == value
        @test isapprox(get_score(new_trace), log(value ? 0.2 : 1 - 0.2))
    end

    @testset "choice_gradients" begin
        y = 1.2
        constraints = choicemap()
        set_value!(constraints, 3, y)
        (trace, _) = generate(choice_at(normal, Int), (0.0, 1.0, 3), constraints)

        # not selected
        (input_grads, choices, gradients) = choice_gradients(
            trace, EmptySelection(), 1.2)
        @test isempty(choices)
        @test isempty(gradients)
        @test length(input_grads) == 3
        @test isapprox(input_grads[1], logpdf_grad(normal, y, 0.0, 1.0)[2])
        @test isapprox(input_grads[2], logpdf_grad(normal, y, 0.0, 1.0)[3])
        @test input_grads[3] == nothing # the key has no gradient

        # selected with retval_grad
        retval_grad = 1.234
        selection = select(3)
        (input_grads, choices, gradients) = choice_gradients(
            trace, selection, retval_grad)
        @test choices[3] == y
        @test isapprox(gradients[3], logpdf_grad(normal, y, 0.0, 1.0)[1] + retval_grad)
        @test length(collect(get_values_shallow(gradients))) == 1
        @test length(collect(get_submaps_shallow(gradients))) == 0
        @test length(collect(get_values_shallow(choices))) == 1
        @test length(collect(get_submaps_shallow(choices))) == 0
        @test length(input_grads) == 3
        @test isapprox(input_grads[1], logpdf_grad(normal, y, 0.0, 1.0)[2])
        @test isapprox(input_grads[2], logpdf_grad(normal, y, 0.0, 1.0)[3])
        @test input_grads[3] == nothing # the key has no gradient
    end

    @testset "accumulate_param_gradients!" begin
        trace = get_trace()
        input_grads = accumulate_param_gradients!(trace, nothing)
        @test length(input_grads) == 2
        @test isapprox(input_grads[1], logpdf_grad(bernoulli, true, 0.4)[2])
        @test input_grads[2] == nothing # the key has no gradient
    end

end
