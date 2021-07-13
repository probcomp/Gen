@testset "custom deterministic generative function with custom update and gradient" begin

    struct MyDeterministicGFState
        sum::Float64
        prev::AbstractArray
        my_param::Float64
    end

    # sum(arr) * my_param
    # arr is the argument
    # my_param is a trainable parameter

    mutable struct MyDeterministicGF <: CustomDetermGF{Float64,MyDeterministicGFState}
        my_param::Float64
        my_grad::Float64
    end

    MyDeterministicGF() = MyDeterministicGF(1., 0.)

    function Gen.apply_with_state(gen_fn::MyDeterministicGF, args::Tuple)
        arr = args[1]
        retval = sum(arr) * gen_fn.my_param
        state = MyDeterministicGFState(retval, arr, gen_fn.my_param)
        retval, state
    end

    function Gen.update_with_state(gen_fn::MyDeterministicGF, state, args, argdiffs)
        arr = args[1]
        retval = sum(arr) * state.my_param
        state = MyDeterministicGFState(retval, arr, state.my_param)
        state, retval, UnknownChange()
    end

    function Gen.update_with_state(::MyDeterministicGF, state, args, argdiffs::Tuple{NoChange})
        state, state.sum * state.my_param, NoChange()
    end

    function Gen.update_with_state(::MyDeterministicGF, state, args, argdiffs::Tuple{VectorDiff})
        arr = args[1]
        retval = state.sum * state.my_param
        for i in keys(argdiffs[1].updated)
            retval += (arr[i] - state.prev[i])
        end
        prev_length = length(state.prev)
        new_length = length(arr)
        for i=prev_length+1:new_length
            retval += arr[i]
        end
        for i=new_length+1:prev_length
            retval -= arr[i]
        end
        state = MyDeterministicGFState(retval, arr, state.my_param)
        state, retval, UnknownChange()
    end

    Gen.has_argument_grads(::MyDeterministicGF) = (true,)

    Gen.accepts_output_grad(::MyDeterministicGF) = true

    function Gen.gradient_with_state(::MyDeterministicGF, state, args, retgrad)
        arr_gradient = fill(retgrad * state.my_param, length(state.prev))
        (arr_gradient,)
    end

    function Gen.accumulate_param_gradients_determ!(
            gen_fn::MyDeterministicGF, state, args, retgrad, scaler)
        gen_fn.my_grad += (retgrad * state.sum) * scaler
        arr_gradient = fill(retgrad * state.my_param, length(state.prev))
        (arr_gradient,)
    end

    # simulate
    gen_fn = MyDeterministicGF()
    trace = simulate(gen_fn, ([1, 2, 3],))
    @test get_retval(trace) == 1 + 2 + 3
    @test get_score(trace) == 0.
    @test project(trace, EmptySelection()) == 0.
    @test isempty(get_choices(trace))
    @test get_args(trace) == ([1, 2, 3],)
    @test get_gen_fn(trace) == gen_fn

    # generate
    trace, w = generate(MyDeterministicGF(), ([1, 2, 3],))
    @test w == 0.
    @test get_retval(trace) == 1 + 2 + 3

    # serialization
    @test serialize_loop_successful(trace)

    # update (UnknownChange)
    trace = simulate(MyDeterministicGF(), ([1, 2, 3],))
    new_trace, w, retdiff = update(trace, ([1, 2, 4],), (UnknownChange(),), EmptyChoiceMap())
    @test w == 0.
    @test get_retval(new_trace) == 1 + 2 + 4
    @test get_args(new_trace) == ([1, 2, 4],)
    @test retdiff == UnknownChange()

    # update (VectorDiff)
    trace = simulate(MyDeterministicGF(), ([1, 2, 3],))
    diff = VectorDiff(4, 3, Dict(3 => UnknownChange()))
    new_trace, w, retdiff = update(trace, ([1, 2, 4, 5],), (diff,), EmptyChoiceMap())
    @test w == 0.
    @test get_retval(new_trace) == 1 + 2 + 4 + 5
    @test get_args(new_trace) == ([1, 2, 4, 5],)
    @test retdiff == UnknownChange()

    # regenerate
    trace = simulate(MyDeterministicGF(), ([1, 2, 3],))
    new_trace, w, retdiff = regenerate(trace, ([1, 2, 4],), (UnknownChange(),), EmptySelection())
    @test w == 0.
    @test get_retval(new_trace) == 1 + 2 + 4
    @test get_args(new_trace) == ([1, 2, 4],)
    @test retdiff == UnknownChange()

    # choice gradients
    trace = simulate(MyDeterministicGF(), ([1, 2, 3],))
    arg_grads, _, _ = choice_gradients(trace, EmptySelection(), 2.)
    @test arg_grads[1] == [2., 2., 2.]

    # accumulate parameter gradients
    gen_fn = MyDeterministicGF()
    trace = simulate(gen_fn, ([1, 2, 3],))
    arg_grads = accumulate_param_gradients!(trace, 2., 3.)
    @test arg_grads[1] == [2., 2., 2.] # not scaled by 3.
    @test gen_fn.my_grad == 2. * (1 + 2 + 3) * 3. # scaled by 3.
end


@testset "custom gradient GF" begin

    struct CustomPlus <: CustomGradientGF{Float64} end
    Gen.has_argument_grads(::CustomPlus) = (true, true)
    Gen.apply(::CustomPlus, args::Tuple{Float64,Float64}) = args[1] + args[2]
    Gen.gradient(::CustomPlus, args, retval, retgrad) = (retgrad, retgrad)
    custom_plus = CustomPlus()
    trace = simulate(custom_plus, (1., 2.))

    # choice gradients
    arg_grads, _, _ = choice_gradients(trace, EmptySelection(), 2.)
    @test arg_grads == (2., 2.)

    # accumulate parameter gradients
    arg_grads = accumulate_param_gradients!(trace, 2., 3.)
    @test arg_grads == (2., 2.) # note arg grads are not scaled
end

@testset "custom update GF" begin

    mutable struct MyState
        prev_arr::Vector{Float64}
        sum::Float64
    end

    struct MySum <: CustomUpdateGF{Float64,MyState} end
    mysum = MySum()

    function Gen.apply_with_state(::MySum, args)
        arr = args[1]
        s = sum(arr)
        state = MyState(arr, s)
        (s, state)
    end

    function Gen.update_with_state(::MySum, state, args, argdiffs::Tuple{VectorDiff})
        arr = args[1]
        prev_sum = state.sum
        retval = prev_sum
        for i in keys(argdiffs[1].updated)
            retval += (arr[i] - state.prev_arr[i])
        end
        prev_length = length(state.prev_arr)
        new_length = length(arr)
        for i=prev_length+1:new_length
            retval += arr[i]
        end
        for i=new_length+1:prev_length
            retval -= arr[i]
        end
        state = MyState(arr, retval)
        (state, retval, UnknownChange())
    end

    Gen.num_args(::MySum) = 1

    # simulate
    trace = simulate(mysum, ([1, 2, 3],))
    @test get_retval(trace) == 1 + 2 + 3
    @test get_score(trace) == 0.

    # update (UnknownChange)
    new_trace, w, retdiff, discard = update(trace, ([1, 2, 4],), (UnknownChange(),), EmptyChoiceMap())
    @test w == 0.
    @test get_retval(new_trace) == 1 + 2 + 4
    @test get_args(new_trace) == ([1, 2, 4],)
    @test retdiff == UnknownChange()
    @test isempty(discard)

    # update (VectorDiff)
    diff = VectorDiff(4, 3, Dict(3 => UnknownChange()))
    new_trace, w, retdiff, discard = update(trace, ([1, 2, 4, 5],), (diff,), EmptyChoiceMap())
    @test w == 0.
    @test get_retval(new_trace) == 1 + 2 + 4 + 5
    @test get_args(new_trace) == ([1, 2, 4, 5],)
    @test retdiff == UnknownChange()
    @test isempty(discard)
end
