using FFTW
using DifferentialEquations


sym_freq_safe(i::Int, n::Int) = n + 2 - i
sym_freq(i::Int, n::Int) = ((i==1) || (i==n÷2+1)) ? i : sym_freq_safe(i, n)


function fourier_xk(L::Float64, n::Int; x0::Float64=0.)
    xx = collect(range(x0; step=L/n, length=n))
    kk = 2 * pi * fftfreq(n, n/L)
    return xx, convert(Vector{Float64}, kk)
end


### 1D ###
function fourier_cmplx2real(ft::Vector{ComplexF64})
    n = length(ft)
    gt = zeros(Float64, n)
    gt[1] = real(ft[1]) # Zero freq
    gt[n÷2+1] = real(ft[n÷2+1]) # Nyquist
    for i = 2:(n÷2)
        j = sym_freq_safe(i, n)
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
        j = sym_freq_safe(i, n)
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
    # Nyquist freq set to zero (arbitrary)
    for i = 2:(n÷2)
        j = sym_freq_safe(i, n)
        s, c = sincos(kk[i] * X)
        gt[i] = c * ft[i] + s * ft[j]
        gt[j] = -s * ft[i] + c * ft[j]
    end
    return gt
end


function derivative_fourier(ft::Vector{Float64}, kk::Vector{Float64})
    n = length(ft)
    gt = zeros(Float64, n)
    for i = 2:(n÷2)
        j = sym_freq_safe(i, n)
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


### 2D ###
struct Iter2D
    n::Int
    m::Int
end

Base.eltype(::Type{Iter2D}) = Tuple{Int, Int, Int, Int}

function Base.iterate(I::Iter2D)
    return (1, 1, 1, 1), (1, 2, I.m÷2+1)
end


"""
```
for i = 1:n
    jmin = (i > n÷2+1) ? 2 : 1
    jmax = m÷2 - jmin + 2
    for j = jmin:jmax
        nothing
    end
end
```
"""
function Base.iterate(I::Iter2D, state::Tuple{Int, Int, Int})
    i, j, jmax = state
    if i > I.n
        return nothing
    end
    if j >= jmax
        in = i + 1
        jn = (in > I.n÷2+1) ? 2 : 1
        jmax = I.m÷2 - jn + 2
    else
        in, jn = i, j+1
    end

    i2, j2 = sym_freq(i, I.n), sym_freq(j, I.m) 
    return ((i, j, i2, j2), (in, jn, jmax))
end


function outer_map(fun, xxs, T=Float64)
    res = zeros(T, (length(xx) for xx in xxs)...)
    for i in CartesianIndices(res)
        res[i] = fun((xx[i[k]] for (k, xx) in enumerate(xxs))...)
    end
    return res
end


function fourier_cmplx2real2(ft::Matrix{ComplexF64})
    s = size(ft) # assume both dimensions are even
    gt = zeros(Float64, s)
    # This is unreadable, and the loops should be swapped for better performance
    for (i, j, i2, j2) in Iter2D(s[1], s[2])
        if (i == i2) && (j == j2) # No symmetric frequency
            gt[i, j] = real(ft[i, j])
        else
            gt[i, j] = sqrt(2.) * real(ft[i, j])  # real part
            gt[i2, j2] = sqrt(2.) * imag(ft[i, j])  # imaginary part
        end
    end
    return gt
end


function fourier_real2cmplx2(ft::Matrix{Float64})
    s = size(ft) # assume both dimensions are even
    gt = zeros(ComplexF64, s)
    # This is unreadable, and the loops should be swapped for better performance
    for (i, j, i2, j2) in Iter2D(s[1], s[2])
        if (i == i2) && (j == j2) # No symmetric frequency
            gt[i, j] = ft[i, j]
        else
            gt[i, j] = (ft[i, j] + 1im * ft[i2, j2]) / sqrt(2.)
            gt[i2, j2] = (ft[i, j] - 1im * ft[i2, j2]) / sqrt(2.)
        end
    end
    return gt
end


function fourier_fun2(fun, kkx::Vector{Float64}, kky::Vector{Float64})
    # Assume that fun(-k) = conj(fun(k))
    s = length(kkx), length(kky) # Assume both dimensions are even
    dx = 2. * pi / (length(kkx) * (kkx[2]-kkx[1])) # spacing in real space
    dy = 2. * pi / (length(kky) * (kky[2]-kky[1])) # spacing in real space
    ft = outer_map(fun, (kkx, kky), ComplexF64) / (dx * dy) # Important factor
    return fourier_cmplx2real2(ft)
end


custom_irfft2(ft) = real(ifft(fourier_real2cmplx2(ft)))


function gaussian_fourier2(
        kkx::Vector{Float64}, kky::Vector{Float64};
        A::Float64=1., x0::Float64=0., y0::Float64=0.,
        σx::Float64=1., σy::Float64=1.)
    a = A * 2*pi * σx * σy
    fun(kx, ky) = a * exp.(-1im*(x0*kx+y0*ky) - (σx*kx)^2/2. - (σy*ky)^2/2.)
    return fourier_fun2(fun, kkx, kky)
end


function shift_fourier2(ft::Matrix{Float64}, kkx::Vector{Float64},
        kky::Vector{Float64}, X::Float64, Y::Float64)
    s = size(ft) # assume both dimensions are even
    gt = zeros(Float64, s)
    for (i, j, i2, j2) in Iter2D(s[1], s[2])
        if (i == i2) && (j == j2) # No symmetric frequency
            # important for zero frequency, arbitrary for Nyquist
            gt[i, j] = ft[i, j]
        else
            si, co = sincos(kkx[i] * X + kky[j] * Y)
            gt[i, j] = co * ft[i, j] + si * ft[i2, j2]
            gt[i2, j2] = -si * ft[i, j] + co * ft[i2, j2]
        end
    end
    return gt
end


function gradient_fourier2(ft::Matrix{Float64}, kkx::Vector{Float64},
        kky::Vector{Float64})
    s = size(ft)
    gt = zeros(Float64, (s[1], s[2], 2))
    for (i, j, i2, j2) in Iter2D(s[1], s[2])
        if !((i == i2) && (j == j2)) # Symmetric frequency
            gt[i, j, 1] = -kkx[i] * ft[i2, j2]
            gt[i, j, 2] = -kky[j] * ft[i2, j2]
            gt[i2, j2, 1] = kkx[i] * ft[i, j]
            gt[i2, j2, 2] = kky[j] * ft[i, j]
        end
    end
    return gt
end


### Numerical integration ###
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

