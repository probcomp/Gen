import DataStructures: OrderedDict

@testset "bernoulli" begin
    # random
    x = bernoulli(0.5)

    # logpdf_grad
    f = (x::Bool, prob::Float64) -> logpdf(bernoulli, x, prob)
    args = (false, 0.3,)
    actual = logpdf_grad(bernoulli, args...)
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
    args = (true, 0.3,)
    actual = logpdf_grad(bernoulli, args...)
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
end

@testset "beta" begin

    # random
    x = beta(0.5, 0.5)
    @test 0 < x < 1

    # out of support
    @test logpdf(beta, -1, 0.5, 0.5) == -Inf

    # logpdf_grad
    f = (x, alpha, beta_param) -> logpdf(beta, x, alpha, beta_param)
    args = (0.4, 0.2, 0.3)
    actual = logpdf_grad(beta, args...)
    @test isapprox(actual[1], finite_diff(f, args, 1, dx))
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
    @test isapprox(actual[3], finite_diff(f, args, 3, dx))
end

@testset "categorical" begin

    # random
    x = categorical([0.2, 0.3, 0.5])
    @test 0 < x < 4

    # out of support
    @test logpdf(categorical, -1, [0.2, 0.3, 0.5]) == -Inf

    # logpdf_grad
    f = (x, probs) -> logpdf(categorical, x, probs)
    args = (2, [0.2, 0.3, 0.5])
    actual = logpdf_grad(categorical, args...)
    @test actual[1] == nothing
    @test isapprox(actual[2][1], finite_diff_vec(f, args, 2, 1, dx))
    @test isapprox(actual[2][2], finite_diff_vec(f, args, 2, 2, dx))
    @test isapprox(actual[2][3], finite_diff_vec(f, args, 2, 3, dx))
end


@testset "gamma" begin

    # random
    x = gamma(1, 1)
    @test 0 < x

    # out of support
    @test logpdf(gamma, -1, 1, 1) == -Inf

    # logpdf_grad
    f = (x, shape, scale) -> logpdf(gamma, x, shape, scale)
    args = (0.4, 0.2, 0.3)
    actual = logpdf_grad(gamma, args...)
    @test isapprox(actual[1], finite_diff(f, args, 1, dx))
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
    @test isapprox(actual[3], finite_diff(f, args, 3, dx))
end

@testset "inv_gamma" begin

    # random
    x = inv_gamma(1, 1)
    @test 0 < x

    # out of support
    @test logpdf(inv_gamma, -1, 1, 1) == -Inf

    # logpdf_grad not implemented

end

@testset "normal" begin
    # random
    x = normal(0, 1)
    # logpdf_grad
    f = (x, mu, std) -> logpdf(normal, x, mu, std)
    args = (0.4, 0.2, 0.3)
    actual = logpdf_grad(normal, args...)
    @test isapprox(actual[1], finite_diff(f, args, 1, dx))
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
    @test isapprox(actual[3], finite_diff(f, args, 3, dx))
end

@testset "zero-dimensional broadcasted normal" begin

    # random
    x = broadcasted_normal(fill(0), fill(1))

    # logpdf_grad
    f = (x, mu, std) -> logpdf(broadcasted_normal, x, mu, std)
    args = (fill(0.4), fill(0.2), fill(0.3))
    actual = logpdf_grad(broadcasted_normal, args...)
    @test isapprox(actual[1], finite_diff(f, args, 1, dx; broadcast=true))
    @test isapprox(actual[2], finite_diff(f, args, 2, dx; broadcast=true))
    @test isapprox(actual[3], finite_diff(f, args, 3, dx; broadcast=true))
end

@testset "array normal (trivially broadcasted: all args have same shape)" begin

    mu =  [ 0.1   0.2   0.3  ;
            0.4   0.5   0.6  ]
    std = [ 0.01  0.02  0.03 ;
            0.04  0.05  0.06 ]
    x =   [ 1.    2.    3.   ;
            4.    5.    6.   ]

    # random
    broadcasted_normal(mu, std)

    # logpdf_grad
    f = (x, mu, std) -> logpdf(broadcasted_normal, x, mu, std)
    args = (mu, std, x)
    actual = logpdf_grad(broadcasted_normal, args...)
    @test isapprox(actual[1], finite_diff(f, args, 1, dx; broadcast=true))
    @test isapprox(actual[2], finite_diff(f, args, 2, dx; broadcast=true))
    @test isapprox(actual[3], finite_diff(f, args, 3, dx; broadcast=true))
end

@testset "broadcasted normal" begin
    ## Return shape of `broadcasted_normal`
    @test size(broadcasted_normal([0. 0. 0.], 1.)) == (1, 3)
    @test size(broadcasted_normal(zeros(1, 3, 4), ones(2, 1, 4))) == (2, 3, 4)
    @test_throws DimensionMismatch broadcasted_normal([0 0 0], [1 1])

    ## Return shape of `logpdf` and `logpdf_grad`
    @test size(logpdf(broadcasted_normal,
                      ones(2, 4), ones(2, 1), ones(1, 4))) == ()
    @test all(size(g) == ()
              for g in logpdf_grad(
                  broadcasted_normal, ones(2, 4), ones(2, 1), ones(1, 4)))
    # `x` has the wrong shape
    @test_throws DimensionMismatch logpdf(broadcasted_normal,
                                          ones(1, 2), ones(1,3), ones(2,1))
    @test_throws DimensionMismatch logpdf_grad(broadcasted_normal,
                                               ones(1, 2), ones(1,3), ones(2,1))
    # `x` has a shape that is broadcast-compatible with but not equal to the
    # right shape
    @test_throws DimensionMismatch logpdf(broadcasted_normal,
                                          ones(1, 3), ones(1,3), ones(2,1))
    @test_throws DimensionMismatch logpdf_grad(broadcasted_normal,
                                               ones(1, 3), ones(1,3), ones(2,1))
    # `mu` and `std` are broadcast-incompatible
    @test_throws DimensionMismatch logpdf(broadcasted_normal,
                                          ones(2, 1), ones(1,2), ones(1,3))
    @test_throws DimensionMismatch logpdf_grad(broadcasted_normal,
                                               ones(2, 1), ones(1,2), ones(1,3))

    ## Equivalence of broadcast to supplying bigger arrays for `mu` and `std`
    compact = OrderedDict(:x => reshape([ 0.2  0.3  0.4  0.5 ;
                                          0.5  0.4  0.3  0.2 ],
                                        (2, 4)),
                          :mu => reshape([0.7  0.7  0.8  0.6],
                                         (1, 4)),
                          :std => reshape([0.2, 0.1],
                                          (2, 1)))
    expanded = OrderedDict(:x => compact[:x],
                           :mu => repeat(compact[:mu], outer=(2, 1)),
                           :std => repeat(compact[:std], outer=(1, 4)))
    @test (logpdf(broadcasted_normal, values(compact)...) ==
           logpdf(broadcasted_normal, values(expanded)...))
    @test (logpdf_grad(broadcasted_normal, values(compact)...) ==
           logpdf_grad(broadcasted_normal, values(expanded)...))
end

@testset "multivariate normal" begin

    # random
    x = mvnormal([0.0, 0.0], [1.0 0.2; 0.2 1.4])
    @test length(x) == 2

    # logpdf_grad
    f = (x, mu, cov) -> logpdf(mvnormal, x, mu, cov)
    args = ([0.1, 0.2], [0.3, 0.4], [1.0 0.2; 0.2 1.4])
    actual = logpdf_grad(mvnormal, args...)
    @test isapprox(actual[1][1], finite_diff_vec(f, args, 1, 1, dx))
    @test isapprox(actual[1][2], finite_diff_vec(f, args, 1, 2, dx))
    @test isapprox(actual[2][1], finite_diff_vec(f, args, 2, 1, dx))
    @test isapprox(actual[2][2], finite_diff_vec(f, args, 2, 2, dx))
    @test isapprox(actual[3][1, 1], finite_diff_mat_sym(f, args, 3, 1, 1, dx))
    @test isapprox(actual[3][1, 2], finite_diff_mat_sym(f, args, 3, 1, 2, dx))
    @test isapprox(actual[3][2, 1], finite_diff_mat_sym(f, args, 3, 2, 1, dx))
    @test isapprox(actual[3][2, 2], finite_diff_mat_sym(f, args, 3, 2, 2, dx))
end

@testset "uniform" begin

    # random
    x = uniform(-0.5, 0.5)
    @test -0.5 < x < 0.5

    # out of support
    @test logpdf(uniform, -1, -0.5, 0.5) == -Inf

    # logpdf_grad
    f = (x, low, high) -> logpdf(uniform, x, low, high)
    args = (0.0, -0.5, 0.5)
    actual = logpdf_grad(uniform, args...)
    @test isapprox(actual[1], finite_diff(f, args, 1, dx))
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
    @test isapprox(actual[3], finite_diff(f, args, 3, dx))
end

@testset "uniform_discrete" begin

    # random
    x = uniform_discrete(10, 20)
    @test 10 <= x <= 20

    # out of support
    @test logpdf(uniform_discrete, -1, 10, 20) == -Inf

    # logpdf_grad
    args = (1, 1, 5)
    actual = logpdf_grad(uniform_discrete, args...)
    @test actual == (nothing, nothing, nothing)
end

@testset "piecewise_uniform" begin

    # random
    x = piecewise_uniform([-0.5, 0.5], [1.0])
    @test -0.5 < x < 0.5

    # out of support
    @test logpdf(piecewise_uniform, -1, [-0.5, 0.5], [1.0]) == -Inf

    # logpdf_grad
    f = (x, bounds, probs) -> logpdf(piecewise_uniform, x, bounds, probs)
    args = (0.5, [-1.0, 0.0, 1.0], [0.4, 0.6])
    actual = logpdf_grad(piecewise_uniform, args...)
    @test isapprox(actual[1], finite_diff(f, args, 1, dx))
    @test isapprox(actual[2][1], finite_diff_vec(f, args, 2, 1, dx))
    @test isapprox(actual[2][2], finite_diff_vec(f, args, 2, 2, dx))
    @test isapprox(actual[2][3], finite_diff_vec(f, args, 2, 3, dx))
    @test isapprox(actual[3][1], finite_diff_vec(f, args, 3, 1, dx))
    @test isapprox(actual[3][2], finite_diff_vec(f, args, 3, 2, dx))
end

@testset "beta uniform mixture" begin

    # random
    x = beta_uniform(0.5, 0.5, 0.5)
    @test 0 < x < 1

    # out of support
    @test logpdf(beta_uniform, -1, 0.5, 0.5, 0.5) == -Inf

    # logpdf_grad
    f = (x, theta, alpha, beta) -> logpdf(beta_uniform, x, theta, alpha, beta)
    args = (0.5, 0.4, 10., 2.)
    actual = logpdf_grad(beta_uniform, args...)
    @test isapprox(actual[1], finite_diff(f, args, 1, dx))
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
    @test isapprox(actual[3], finite_diff(f, args, 3, dx))
    @test isapprox(actual[4], finite_diff(f, args, 4, dx))
end

@testset "geometric" begin

    # random
    @test geometric(0.5) >= 0

    # out of support
    @test logpdf(geometric, -1, 0.5) == -Inf

    # logpdf_grad
    f = (x, p) -> logpdf(geometric, x, p)
    args = (4, 0.3)
    actual = logpdf_grad(geometric, args...)
    @test actual[1] == nothing
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
end

@testset "binom" begin

    # random
    @test binom(5, 0.3) >= 0

    # out of support
    @test logpdf(binom, -1, 5, 0.3) == -Inf

    # logpdf_grad
    f = (x, n, p) -> logpdf(binom, x, n, p)
    args = (2, 5, 0.3)
    actual = logpdf_grad(binom, args...)
    @test actual[1] == nothing
    @test actual[2] == nothing
    @test isapprox(actual[3], finite_diff(f, args, 3, dx))
end

@testset "neg_binom" begin

    # random
    @test neg_binom(5, 0.3) >= 0

    # out of support
    @test logpdf(neg_binom, -1, 5, 0.3) == -Inf

    # logpdf_grad
    f = (x, r, p) -> logpdf(neg_binom, x, r, p)
    args = (2, 5, 0.3)
    actual = logpdf_grad(neg_binom, args...)
    @test actual[1] == nothing
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
    @test isapprox(actual[3], finite_diff(f, args, 3, dx))
end

@testset "exponential" begin

    # random
    @test exponential(0.5) > 0

    # out of support
    @test logpdf(exponential, -1, 0.5) == -Inf

    # logpdf_grad
    f = (x, rate) -> logpdf(exponential, x, rate)
    args = (1.2, 0.5)
    actual = logpdf_grad(exponential, args...)
    @test isapprox(actual[1], finite_diff(f, args, 1, dx))
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
end

@testset "poisson" begin

    # random
    @test poisson(1.0) >= 0

    # out of support
    @test logpdf(poisson, -1, 1.0) == -Inf

    # logpdf_grad
    f = (x, lambda) -> logpdf(poisson, x, lambda)
    args = (4, 2.0)
    actual = logpdf_grad(poisson, args...)
    @test actual[1] == nothing
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
end

@testset "laplace" begin

    # random
    x = laplace(1, 1)

    # logpdf_grad
    f = (x, loc, scale) -> logpdf(laplace, x, loc, scale)
    args = (0.4, 0.2, 0.3)
    actual = logpdf_grad(laplace, args...)
    @test isapprox(actual[1], finite_diff(f, args, 1, dx))
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
    @test isapprox(actual[3], finite_diff(f, args, 3, dx))
end


@testset "dirichlet" begin
    # random
    alpha = [2.0, 2.0, 2.0]
    x = dirichlet(alpha)
    f = (x, alpha) -> logpdf(dirichlet, x, alpha)
    # args = (x, alpha) where x has to be simplex
    args = ([0.05,0.25,0.7], alpha)
    actual = logpdf_grad(dirichlet, args...) # gradient wrt x, then alpha1 ... alphak
    @test isapprox(actual[2], finite_diff_vec(f, args, 2, 1, dx), rtol=1e-6)
    @test isapprox(actual[3], finite_diff_vec(f, args, 2, 2, dx), rtol=1e-6)
    @test isapprox(actual[4], finite_diff_vec(f, args, 2, 3, dx), rtol=1e-6)

    x = dirichlet(alpha)
    args = (x, alpha)
    actual = logpdf_grad(dirichlet, args...) # gradient wrt x, then alpha1 ... alphak
    @test isapprox(actual[2], finite_diff_vec(f, args, 2, 1, dx), rtol=1e-6)
    @test isapprox(actual[3], finite_diff_vec(f, args, 2, 2, dx), rtol=1e-6)
    @test isapprox(actual[4], finite_diff_vec(f, args, 2, 3, dx), rtol=1e-6)

    x = dirichlet(alpha)
    args = (x, alpha)
    actual = logpdf_grad(dirichlet, args...) # gradient wrt x, then alpha1 ... alphak
    @test isapprox(actual[2], finite_diff_vec(f, args, 2, 1, dx), rtol=1e-6)
    @test isapprox(actual[3], finite_diff_vec(f, args, 2, 2, dx), rtol=1e-6)
    @test isapprox(actual[4], finite_diff_vec(f, args, 2, 3, dx), rtol=1e-6)
end
