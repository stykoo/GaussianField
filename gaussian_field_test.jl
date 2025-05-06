using Test
include("gaussian_field.jl")

@testset "Fourier 1d test" begin
    L, n = 10., 128
    dx = L/n
    A, x0, σ = 2.2, 4.5, 0.75
    X = 0.4
    atol, rtol = 1e-7, 1e-5
    atol2, rtol2 = 1e-3, 1e-3 # not precise but OK

    # Complex / real conversion
    rdata = randn(Float64, n)
    fdata = fft(rdata)
    @test isapprox(fourier_cmplx2real(fourier_real2cmplx(rdata)),
                   rdata; atol=atol, rtol=rtol)
    @test isapprox(fourier_real2cmplx(fourier_cmplx2real(fdata)),
                   fdata; atol=atol, rtol=rtol)

    # Standard Fourier transform
    xx, kk = fourier_xk(L, n)
    yy = exp.(-(xx.-x0) .^2 / 2.) / sqrt(2. * pi)
    fun(k) = exp.(-1im*x0*k - k.^2/ 2.)

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

@testset "Fourier 2d test" begin
    Lx, nx = 10., 16
    Ly, ny = 6., 8
    dx, dy = Lx/nx, Ly/ny
    A = 1.4
    σx, σy = 0.9, 1.2
    x0, y0 = 4.5, 2.8
    X, Y = 0.4, -0.7
    atol, rtol = 1e-7, 1e-5
    atol2, rtol2 = 1e-5, 0.02 # rtol is high

    # Complex / real conversion
    rdata = randn(Float64, (nx, ny))
    fdata = fft(rdata)
    @test isapprox(fourier_cmplx2real2(fourier_real2cmplx2(rdata)),
                   rdata; atol=atol, rtol=rtol)
    @test isapprox(fourier_real2cmplx2(fourier_cmplx2real2(fdata)),
                   fdata; atol=atol, rtol=rtol)

    xx, kkx = fourier_xk(Lx, nx)
    yy, kky = fourier_xk(Ly, ny)
    fun_r(x, y) = exp.(-((x-x0)^2+(y-y0)^2)/2.) / (2. * pi)
    fun_k(kx, ky) = exp.(-1im*(x0*kx + y0*ky) - (kx^2+ky^2)/2.)
    zz = outer_map(fun_r, (xx, yy))
    zzf0 = outer_map(fun_k, (kkx, kky), ComplexF64)

    # Standard Fourier transform
    @test isapprox(dx*dy*fft(zz), zzf0; atol=atol2, rtol=rtol2)
    @test isapprox(zz, real(ifft(zzf0))/(dx*dy); atol=atol2, rtol=rtol2)

    # Custom Fourier transform
    zzf = fourier_fun2(fun_k, kkx, kky)
    @test isapprox(custom_irfft2(zzf), zz; atol=atol2, rtol=rtol2)
    @test isapprox(custom_irfft2(zzf), zz; atol=atol2, rtol=rtol2)

    # Gaussian potential
    fun_g(x, y) = A .* exp.(-(x-x0)^2/(2*σx^2)-(y-y0)^2/(2*σy^2))
    gg = outer_map(fun_g, (xx, yy))
    ggf = gaussian_fourier2(kkx, kky; A=A, x0=x0, y0=y0, σx=σx, σy=σy)
    @test isapprox(custom_irfft2(ggf), gg; atol=atol2, rtol=rtol2)

    # Shift
    zzs = outer_map(fun_r, (xx.-X, yy.-Y))
    zzfs = shift_fourier2(zzf, kkx, kky, X, Y)
    @test isapprox(custom_irfft2(zzfs), zzs; atol=atol2, rtol=rtol2)
    
    # Gradient
    zzd = zeros(nx, ny, 2)
    for j=1:ny, i=1:nx
        zzd[i, j, 1] = -(xx[i] - x0) * zz[i, j]
        zzd[i, j, 2] = -(yy[j] - y0) * zz[i, j]
    end
    zzfd = gradient_fourier2(zzf, kkx, kky)
    @test isapprox(custom_irfft2(zzfd[:, :, 1]), zzd[:, :, 1]; atol=atol2, rtol=rtol2)
    @test isapprox(custom_irfft2(zzfd[:, :, 2]), zzd[:, :, 2]; atol=atol2, rtol=rtol2)

    # Integration
    I1 = dx * dy * sum(zz .^2)
    I1f = integrate_plancherel_fourier2(zzf, zzf, kkx, kky)
    @test isapprox(I1, I1f; atol=atol2, rtol=rtol2)
    I2 = dx * dy * sum(zz .* gg)
    I2f = integrate_plancherel_fourier2(zzf, ggf, kkx, kky)
    @test isapprox(I2, I2f; atol=atol2, rtol=rtol2)
end
