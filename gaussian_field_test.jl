using Test
include("gaussian_field.jl")

@testset "Fourier test" begin
    L, n = 10., 128
    atol, rtol = 1e-7, 1e-5
    atol2, rtol2 = 1e-3, 1e-3 # not precise but OK
    dx = L/n
    A, x0, σ = 2.2, 4.5, 0.75
    X = 0.4

    xx, kk = fourier_xk(L, n)
    yy = exp.(-(xx.-x0) .^2 / 2.) / sqrt(2. * pi)
    fun(k) = exp.(-1im*x0*k - k.^2/ 2.)

    # Standard Fourier transform
    @test isapprox(dx*fft(yy), fun(kk); atol=atol, rtol=rtol)
    @test isapprox(yy, real(ifft(fun(kk)))/dx; atol=atol, rtol=rtol)

    # Custom Fourier transform
    yyf = fourier_fun(fun, kk)
    @test isapprox(custom_irfft(yyf), yy; atol=atol, rtol=rtol)

    # Gaussian potential
    gg = A .* exp.(-(xx.-x0).^2 ./ (2*σ^2))
    ggf = gaussian_fourier(kk; A=A, x0=x0, σ=σ)
    @test isapprox(custom_irfft(ggf), gg; atol=atol, rtol=rtol)

    # Shift
    yys = exp.(-(xx.-x0.-X) .^2 / 2.) / sqrt(2. * pi)
    yyfs = shift_fourier(yyf, kk, X)
    @test isapprox(custom_irfft(yyfs), yys; atol=atol2, rtol=rtol2)
    
    # Derivative
    yyd = -(xx.-x0) .* yy
    yyfd = derivative_fourier(yyf, kk)
    @test isapprox(custom_irfft(yyfd), yyd; atol=atol2, rtol=rtol2)

    # Integration
    I1 = dx * sum(yy .^2)
    I1f = integrate_plancherel_fourier(yyf, yyf, kk)
    @test isapprox(I1, I1f; atol=atol, rtol=rtol)
    I2 = dx * sum(yy .* gg)
    I2f = integrate_plancherel_fourier(yyf, ggf, kk)
    @test isapprox(I2, I2f; atol=atol, rtol=rtol)
end
