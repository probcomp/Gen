@testset "incremental computation for user-defined structs" begin

    struct Point2D{T <: Real}
        x::T
        y::T
    end

    Point2D(x) = Point2D(x, x)

    @testset "diff structs" begin
        diff = StructDiff{Point2D{Float64}}(NoChange(), UnknownChange())
        @test Gen.get_field_diff(diff, :x) == NoChange()
        @test Gen.get_field_diff(diff, :y) == UnknownChange()
    end

    @testset "type construction" begin
        gen_fn = Construct(Point2D)
        trace = simulate(gen_fn, (1, 2))
        @test get_retval(trace) == Point2D(1, 2)

        # Expect NoChange() if no fields change
        new_trace, _, retdiff, _ =
            update(trace, (1, 2), (NoChange(), NoChange()), choicemap())
        @test get_retval(new_trace) == Point2D(1, 2)
        @test retdiff == NoChange()

        # Expect UnknownChange() if all fields have unknown changes
        new_trace, _, retdiff, _ =
            update(trace, (4, 3), (UnknownChange(), UnknownChange()), choicemap())
        @test get_retval(new_trace) == Point2D(4, 3)
        @test retdiff == UnknownChange()

        # Expect a StructDiff if fields change
        new_trace, _, retdiff, _ =
            update(trace, (4, 3), (IntDiff(3), UnknownChange()), choicemap())
        @test get_retval(new_trace) == Point2D(4, 3)
        @test retdiff == StructDiff{Point2D}(IntDiff(3), UnknownChange())

        # Expect UnknownChange() if a non-default constructor is used
        new_trace, _, retdiff, _ =
            update(trace, (0,), (UnknownChange(),), choicemap())
        @test get_retval(new_trace) == Point2D(0, 0)
        @test retdiff == UnknownChange()
    end

    @testset "field access" begin
        point = Point2D(1, 2)

        get_x = GetField(Point2D, :x)
        get_y = GetField(Point2D, :y)
        trace_x = simulate(get_x, (point,))
        trace_y = simulate(get_y, (point,))
        @test get_retval(trace_x) == 1
        @test get_retval(trace_y) == 2

        # Expect NoChange() if no fields change
        new_trace_x, _, retdiff_x, _ =
            update(trace_x, (point,), (NoChange(),), choicemap())
        new_trace_y, _, retdiff_y, _ =
            update(trace_y, (point,), (NoChange(),), choicemap())
        @test get_retval.((new_trace_x, new_trace_y)) == (1, 2)
        @test (retdiff_x, retdiff_y) == (NoChange(), NoChange())

        # Expect StructDiff changes to propagate
        new_point = Point2D(0, 4)
        diff = StructDiff{Point2D}(UnknownChange(), IntDiff(2))
        new_trace_x, _, retdiff_x, _ =
            update(trace_x, (new_point,), (diff,), choicemap())
        new_trace_y, _, retdiff_y, _ =
            update(trace_y, (new_point,), (diff,), choicemap())
        @test get_retval.((new_trace_x, new_trace_y)) == (0, 4)
        @test (retdiff_x, retdiff_y) == (UnknownChange(), IntDiff(2))

        # Expect UnknownChange() if argdiff is UnknownChange()
        new_trace_x, _, retdiff_x, _ =
            update(trace_x, (new_point,), (UnknownChange(),), choicemap())
        new_trace_y, _, retdiff_y, _ =
            update(trace_y, (new_point,), (UnknownChange(),), choicemap())
        @test get_retval.((new_trace_x, new_trace_y)) == (0, 4)
        @test (retdiff_x, retdiff_y) == (UnknownChange(), UnknownChange())
    end

    @testset "test function" begin
        @gen (static,diffs) function foo()
            x ~ normal(0, 1)
            y ~ normal(0, 1)
            p ~ Construct(Point2D{Float64})(x, y)
            a ~ GetField(Point2D{Float64}, :x)(p)
            b ~ GetField(Point2D{Float64}, :y)(p)
            q ~ Construct(Point2D{Float64})(a, b)
            return p
        end

        Gen.load_generated_functions(@__MODULE__)

        trace, _ = generate(foo, (), choicemap(:x => 1, :y => 2))
        @test trace[:p] == Point2D{Float64}(1, 2)
        @test trace[:a] == 1
        @test trace[:b] == 2
        @test get_retval(trace) == Point2D{Float64}(1, 2)

        # Expect NoChange() if choices are not updated
        new_trace, _, retdiff, _ = update(trace, (), (), choicemap())
        @test get_retval(new_trace) == Point2D{Float64}(1, 2)
        @test retdiff == NoChange()

        # Expect changes to each field to propagate separately
        new_trace, _, retdiff, _ = update(trace, (), (), choicemap(:y => 1))
        @test get_retval(new_trace) == Point2D{Float64}(1, 1)
        @test retdiff == StructDiff{Point2D{Float64}}(NoChange(), UnknownChange())

        # Expect UnknownChange() if all fields have unknown changes
        new_trace, _, retdiff, _ =
            update(trace, (), (), choicemap(:x => 0, :y => 1))
        @test get_retval(new_trace) == Point2D{Float64}(0, 1)
        @test retdiff == UnknownChange()
    end

end
