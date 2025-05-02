using FFTW
using DifferentialEquations


function fourier_xk(L::Float64, n::Int; x0::Float64=0.)
    xx = collect(range(x0; step=L/n, length=n))
    kk = 2 * pi * fftfreq(n, n/L)
    return xx, convert(Vector{Float64}, kk)
end


function fourier_cmplx2real(ft::Vector{ComplexF64})
    n = length(ft)
    gt = zeros(Float64, n)
    gt[1] = real(ft[1]) # Zero freq
    gt[n÷2+1] = real(ft[n÷2+1]) # Nyquist
    for i = 2:(n÷2)
        j = n + 2 - i
        gt[i] = sqrt(2) * real(ft[i])
        gt[j] = sqrt(2) * imag(ft[i])
    end
    return gt
end


function fourier_real2cmplx(ft::Vector{Float64})
    n = length(ft)
    gt = zeros(ComplexF64, n)
    gt[1] = ft[1] # Zero freq
    gt[n÷2+1] = ft[n÷2+1] # Nyquist
    for i = 2:(n÷2)
        j = n + 2 - i
        gt[i] = (ft[i] + 1im * ft[j]) / sqrt(2.)
        gt[j] = (ft[i] - 1im * ft[j]) / sqrt(2.)
    end
    return gt
end


function fourier_fun(fun, kk::Vector{Float64})
    # Assume that fun(-k) = conj(fun(k))
    dx = 2. * pi / (length(kk) * (kk[2]-kk[1])) # spacing in real space
    ft = fun(kk) / dx # Important factor
    return fourier_cmplx2real(ft)
end


"""
    custom_irfft(ft::Vector{Float64})

Compute the inverse discrete Fourier transform of a real
signal where the +k and -k components encode respectively
the real and imaginary parts of the modes.
"""
custom_irfft(ft::Vector{Float64}) = real(ifft(fourier_real2cmplx(ft)))


function gaussian_fourier(kk::Vector{Float64}; A::Float64=1.,
                          x0::Float64=0., σ::Float64=1.)
    fun(k) = A * sqrt(2*pi) * σ * exp.(-1im*x0.*k - (σ.*k).^2/2.)
    return fourier_fun(fun, kk)
end


function shift_fourier(ft::Vector{Float64}, kk::Vector{Float64}, X::Float64)
    # In ft, the real part is encoded as +k component
    # and the imaginary part as the -k component.
    # Assuming an even number of modes
    n = length(ft)
    gt = zeros(Float64, n)
    gt[1] = ft[1] # Zero freq
    for i = 2:(n÷2)
        j = n + 2 - i
        s, c = sincos(kk[i] * X)
        gt[i] = c * ft[i] + s * ft[j]
        gt[j] = -s * ft[i] + c * ft[j]
    end
    # Nyquist freq component remains zero
    return gt
end


function derivative_fourier(ft::Vector{Float64}, kk::Vector{Float64})
    n = length(ft)
    gt = zeros(Float64, n)
    for i = 2:(n÷2)
        j = n + 2 - i
        gt[i] = -kk[i] * ft[j]
        gt[j] = kk[i] * ft[i]
    end
    # Nyquist freq component remains zero
    return gt
end


function integrate_plancherel_fourier(ft::Vector{Float64}, gt::Vector{Float64},
                                      kk::Vector{Float64})
    # dx = 2. * pi / (length(kk) * dk) # spacing in real space
    # fac = dx^2 * dk / (2 * pi)
    dk = kk[2] - kk[1]
    fac = (2. * pi) / (length(kk)^2 * dk)
    s = sum(ft .* gt) # Check that this gives the real part of the integral
    return fac * s
end


function outer_fun(fun, xxs, T=Float64)
    res = zeros(T, (length(xx) for xx in xxs)...)
    for i in CartesianIndices(res)
        res[i] = fun((xx[i[k]] for (k, xx) in enumerate(xxs))...)
    end
    return res
end


function fourier_cmplx2real2(ft::Matrix{ComplexF64})
    s = size(ft) # assume both dimensions are even
    gt = zeros(Float64, s)
    for j = 1:(s[2]÷2+1), i = 1:(s[1]÷2+1)
        i2 = ((i == 1) || (i == s[1]÷2+1)) ? i : s[1] + 2 - i
        j2 = ((j == 1) || (j == s[2]÷2+1)) ? j : s[2] + 2 - j
        if (i == i2) && (j == j2) # No symmetric frequency
            gt[i, j] = real(ft[i, j])
        else
            gt[i, j] = sqrt(2.) * real(ft[i, j])  # real part
            gt[i2, j2] = sqrt(2.) * imag(ft[i, j])  # imaginary part
        end
    end
    for j = 2:(s[2]÷2), i=(s[1]÷2+2):s[1]
        i2 = s[1] + 2 - i
        j2 = s[2] + 2 - j
        gt[i, j] = sqrt(2.) * real(ft[i, j])  # real part
        gt[i2, j2] = sqrt(2.) * imag(ft[i, j])  # imaginary part
    end
    return gt
end


function fourier_real2cmplx2(ft::Matrix{Float64})
    s = size(ft) # assume both dimensions are even
    gt = zeros(ComplexF64, s)
    for j = 1:(s[2]÷2+1), i = 1:(s[1]÷2+1)
        i2 = ((i == 1) || (i == s[1]÷2+1)) ? i : s[1] + 2 - i
        j2 = ((j == 1) || (j == s[2]÷2+1)) ? j : s[2] + 2 - j
        if (i == i2) && (j == j2) # No symmetric frequency
            gt[i, j] = ft[i, j]
        else
            gt[i, j] = (ft[i, j] + 1im * ft[i2, j2]) / sqrt(2.)
            gt[i2, j2] = (ft[i, j] - 1im * ft[i2, j2]) / sqrt(2.)
        end
    end
    for j = 2:(s[2]÷2), i=(s[1]÷2+2):s[1]
        i2 = s[1] + 2 - i
        j2 = s[2] + 2 - j
        gt[i, j] = (ft[i, j] + 1im * ft[i2, j2]) / sqrt(2.)
        gt[i2, j2] = (ft[i, j] - 1im * ft[i2, j2]) / sqrt(2.)
    end
    return gt
end


function fourier_fun2(fun, kkx::Vector{Float64}, kky::Vector{Float64})
    # Assume that fun(-k) = conj(fun(k))
    s = length(kkx), length(kky) # Assume both dimensions are even
    dx = 2. * pi / (length(kkx) * (kkx[2]-kkx[1])) # spacing in real space
    dy = 2. * pi / (length(kky) * (kky[2]-kky[1])) # spacing in real space

    ft = outer_fun(fun, (kkx, kky), ComplexF64) / (dx * dy) # Important factor
    return fourier_cmplx2real2(ft)
end


custom_irfft2(ft) = real(ifft(fourier_real2cmplx2(ft)))


function sde_drift!(du, u, P, t) 
    kk, VV, p = P
    n = length(kk)
    m = length(du) - n
    XX = u[n+1:end]
    VVs = [shift_fourier(VV, kk, X) for X in XX] # Shifted potentials

    # Field dynamics
    for i = 1:n
        w = (p.r + kk[i]^2) * u[i]
        for j = 1:m
            w -= VVs[j][i]
        end
        du[i] = -p.D * kk[i]^2 * w
    end

    # Particle dynamics
    for j = 1:m
        f = -integrate_plancherel_fourier(
            u[1:n], derivative_fourier(VVs[j], kk), kk
        ) 
        # take initial position as center of trap 
        du[n+j] = -p.k * (XX[j] - p.X0[j]) + f
    end
end


function sde_diff!(du, u, P, t)
    kk, VV, p = P
    n = length(kk)
    for i = 1:n
        du[i] = 2. * p.T * p.D * kk[i]
    end
    du[n+1:end] .= 2 * p.T
end


function run(p; solver=ImplicitRKMil(autodiff=AutoFiniteDiff()))
    # No support for odd number of divisions
    @assert (p.n % 2 == 0)
    xx, kk = fourier_xk(p.L, p.n)
    # Potential in Fourier space
    VVk = gaussian_fourier(kk; A=p.A, σ=p.σ, x0=0.)
    # Initial condition in Fourier space, last indice is tracer position
    u0 = zeros(Float64, p.n+p.m)
    u0[p.n+1:end] = p.X0
    prob = SDEProblem(sde_drift!, sde_diff!, u0, (0.0, p.tmax), (kk, VVk, p))
    sol = solve(prob, solver; saveat=p.saveat)
    return sol
end

