using Gen: get_child, get_child_num, get_parent

@testset "recurse node numbering" begin

    # max_branch = 1
    @test get_child(1, 1, 1) == 2
    @test get_child(2, 1, 1) == 3
    @test get_child(3, 1, 1) == 4
    @test get_child_num(2, 1) == 1
    @test get_child_num(3, 1) == 1
    @test get_child_num(4, 1) == 1
    @test get_parent(2, 1) == 1
    @test get_parent(3, 1) == 2
    @test get_parent(4, 1) == 3

    # max_branch = 2
    @test get_child(1, 1, 2) == 2
    @test get_child(1, 2, 2) == 3
    @test get_child(2, 1, 2) == 4
    @test get_child(2, 2, 2) == 5
    @test get_child(3, 1, 2) == 6
    @test get_child_num(2, 2) == 1
    @test get_child_num(3, 2) == 2
    @test get_child_num(4, 2) == 1
    @test get_child_num(5, 2) == 2
    @test get_child_num(6, 2) == 1
    @test get_parent(2, 2) == 1
    @test get_parent(3, 2) == 1
    @test get_parent(4, 2) == 2
    @test get_parent(5, 2) == 2
    @test get_parent(6, 2) == 3

    # max_branch = 3
    @test get_child(1, 1, 3) == 2
    @test get_child(1, 2, 3) == 3
    @test get_child(1, 3, 3) == 4
    @test get_child(2, 1, 3) == 5
    @test get_child(2, 2, 3) == 6
    @test get_child(2, 3, 3) == 7
    @test get_child(3, 1, 3) == 8
    @test get_child(3, 2, 3) == 9
    @test get_child(3, 3, 3) == 10
    @test get_child(4, 1, 3) == 11
    @test get_child_num(2, 3) == 1
    @test get_child_num(3, 3) == 2
    @test get_child_num(4, 3) == 3
    @test get_child_num(5, 3) == 1
    @test get_child_num(6, 3) == 2
    @test get_child_num(7, 3) == 3
    @test get_child_num(8, 3) == 1
    @test get_child_num(9, 3) == 2
    @test get_child_num(10, 3) == 3
    @test get_child_num(11, 3) == 1
    @test get_parent(2, 3) == 1
    @test get_parent(3, 3) == 1
    @test get_parent(4, 3) == 1
    @test get_parent(5, 3) == 2
    @test get_parent(6, 3) == 2
    @test get_parent(7, 3) == 2
    @test get_parent(8, 3) == 3
    @test get_parent(9, 3) == 3
    @test get_parent(10, 3) == 3
    @test get_parent(11, 3) == 4
end

@testset "simple pcfg" begin

    @gen function pcfg_production(_::Nothing)
        production_rule = @trace(categorical([0.24, 0.26, 0.23, 0.27]), :rule)
        if production_rule == 1
            # aSa; one child
            num_children = 1
        elseif production_rule == 2
            # bSb; one child
            num_children = 1
        elseif production_rule == 3
            # aa; no child
            num_children = 0
        else
            # bb; no child
            num_children = 0
        end

        return Production(production_rule, [nothing for _=1:num_children])
    end

    @gen function pcfg_aggregation(production_rule::Int, child_outputs::Vector{String})
        prefix = @trace(bernoulli(0.4), :prefix) ? "." : "-"
        local str::String
        if production_rule == 1
            @assert length(child_outputs) == 1
            str = "($(prefix)a$(child_outputs[1])a)"
        elseif production_rule == 2
            @assert length(child_outputs) == 1
            str = "($(prefix)b$(child_outputs[1])b)"
        elseif production_rule == 3
            @assert length(child_outputs) == 0
            str = "($(prefix)aa)"
        else
            @assert length(child_outputs) == 0
            str = "($(prefix)bb)"
        end
        return str
    end

    pcfg = Recurse(pcfg_production, pcfg_aggregation, 1, Nothing, Int, String)

    # test that each of the most probable strings are all produced
    function test_strings(strings)
        @test "(.aa)" in strings
        @test "(-aa)" in strings
        @test "(.bb)" in strings
        @test "(-bb)" in strings
        @test "(.a(.aa)a)" in strings
        @test "(-a(.aa)a)" in strings
        @test "(.a(-aa)a)" in strings
        @test "(-a(-aa)a)" in strings
        @test "(.b(.bb)b)" in strings
        @test "(-b(.bb)b)" in strings
        @test "(.b(-bb)b)" in strings
        @test "(-b(-bb)b)" in strings
        @test "(.a(.bb)a)" in strings
        @test "(-a(.bb)a)" in strings
        @test "(.a(-bb)a)" in strings
        @test "(-a(-bb)a)" in strings
        @test "(.b(.aa)b)" in strings
        @test "(-b(.aa)b)" in strings
        @test "(.b(-aa)b)" in strings
        @test "(-b(-aa)b)" in strings
    end

    # test Julia call
    @test isa(pcfg(nothing, 1), String)

    # test generate
    Random.seed!(1)
    strings = Set{String}()
    for i=1:1000
        (trace, _) = generate(pcfg, (nothing, 1), EmptyChoiceMap())
        push!(strings, get_retval(trace))
    end
    test_strings(strings)

    # test simulate
    Random.seed!(1)
    strings = Set{String}()
    for i=1:1000
        trace = simulate(pcfg, (nothing, 1))
        push!(strings, get_retval(trace))
    end
    test_strings(strings)

    # apply generate to a complete choice map that produces "(.b(.a(.b(-bb)b)a)b)"
    # sequence of production rules: 2 -> 1 -> 2 -> 4
    expected_weight = log(0.26) + log(0.24) + log(0.26) + log(0.27) + log(0.4) + log(0.4) + log(0.4) + log(0.6)
    constraints = choicemap()
    constraints[(1, Val(:production)) => :rule] = 2
    constraints[(1, Val(:aggregation)) => :prefix] = true
    constraints[(2, Val(:production)) => :rule] = 1
    constraints[(2, Val(:aggregation)) => :prefix] = true
    constraints[(3, Val(:production)) => :rule] = 2
    constraints[(3, Val(:aggregation)) => :prefix] = true
    constraints[(4, Val(:production)) => :rule] = 4
    constraints[(4, Val(:aggregation)) => :prefix] = false
    (trace, actual_weight) = generate(pcfg, (nothing, 1), constraints)
    @test isapprox(actual_weight, expected_weight)
    @test isapprox(get_score(trace), actual_weight)
    @test get_args(trace) == (nothing, 1)
    @test get_retval(trace) == "(.b(.a(.b(-bb)b)a)b)"
    choices = get_choices(trace)
    @test choices[(1, Val(:production)) => :rule] == 2
    @test choices[(1, Val(:aggregation)) => :prefix] == true
    @test choices[(2, Val(:production)) => :rule] == 1
    @test choices[(2, Val(:aggregation)) => :prefix] == true
    @test choices[(3, Val(:production)) => :rule] == 2
    @test choices[(3, Val(:aggregation)) => :prefix] == true
    @test choices[(4, Val(:production)) => :rule] == 4
    @test choices[(4, Val(:aggregation)) => :prefix] == false

    @test serialize_loop_successful(trace)

    # update non-structure choice
    new_constraints = choicemap()
    new_constraints[(3, Val(:aggregation)) => :prefix] = false
    (new_trace, weight, retdiff, discard) = update(
        trace, (nothing, 1), (UnknownChange(), UnknownChange()), new_constraints)
    @test isapprox(weight, log(0.6) - log(0.4))
    expected_score = log(0.26) + log(0.24) + log(0.26) + log(0.27) + log(0.4) + log(0.4) + log(0.6) + log(0.6)
    @test isapprox(get_score(new_trace), expected_score)
    @test get_args(new_trace) == (nothing, 1)
    @test get_retval(new_trace) == "(.b(.a(-b(-bb)b)a)b)"
    choices = get_choices(new_trace)
    @test choices[(1, Val(:production)) => :rule] == 2
    @test choices[(1, Val(:aggregation)) => :prefix] == true
    @test choices[(2, Val(:production)) => :rule] == 1
    @test choices[(2, Val(:aggregation)) => :prefix] == true
    @test choices[(3, Val(:production)) => :rule] == 2
    @test choices[(3, Val(:aggregation)) => :prefix] == false
    @test choices[(4, Val(:production)) => :rule] == 4
    @test choices[(4, Val(:aggregation)) => :prefix] == false
    @test discard[(3, Val(:aggregation)) => :prefix] == true
    @test length(collect(get_submaps_shallow(discard))) == 1
    @test length(collect(get_values_shallow(discard))) == 0
    @test length(collect(get_submaps_shallow(get_submap(discard,(3, Val(:aggregation)))))) == 0
    @test length(collect(get_values_shallow(get_submap(discard,(3, Val(:aggregation)))))) == 1
    @test retdiff == UnknownChange()

    # update structure choice, so that string becomes: (.b(.a(.aa)a)b)
    # note: we reuse the prefix choice from node 3 (true)
    new_constraints = choicemap()
    new_constraints[(3, Val(:production)) => :rule] = 3 # change from rule 2 to rule 3
    (new_trace, weight, retdiff, discard) = update(
        trace, (nothing, 1), (UnknownChange(), UnknownChange()), new_constraints)
    @test isapprox(weight, log(0.23) - log(0.26) - log(0.27) - log(0.6))
    @test isapprox(get_score(new_trace), log(0.26) + log(0.24) + log(0.23) + log(0.4) + log(0.4) + log(0.4))
    @test get_args(new_trace) == (nothing, 1)
    @test get_retval(new_trace) == "(.b(.a(.aa)a)b)"
    choices = get_choices(new_trace)
    @test choices[(1, Val(:production)) => :rule] == 2
    @test choices[(1, Val(:aggregation)) => :prefix] == true
    @test choices[(2, Val(:production)) => :rule] == 1
    @test choices[(2, Val(:aggregation)) => :prefix] == true
    @test choices[(3, Val(:production)) => :rule] == 3
    @test choices[(3, Val(:aggregation)) => :prefix] == true
    @test isempty(get_submap(choices, (4, Val(:production)))) # FAIL
    @test isempty(get_submap(choices, (4, Val(:aggregation))))
    @test discard[(3, Val(:production)) => :rule] == 2
    @test !has_value(discard, (3, Val(:aggregation)) => :prefix)
    @test discard[(4, Val(:production)) => :rule] == 4
    @test discard[(4, Val(:aggregation)) => :prefix] == false
    @test retdiff == UnknownChange()

end
