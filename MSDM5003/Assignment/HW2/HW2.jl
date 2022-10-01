# mute logging in jupyter notebook
import Logging
Logging.disable_logging(Logging.Warn)

# load dependency
using Distributions, Plots, StatsPlots

function randwalk(X₀::Vector{<:Real},T::Real;σ²=2,dUₓ=x->0,Δt=0.1)
    Δtξ = Normal(0,√(σ²*Δt))
    N = length(X₀)
    ts = 0:Δt:T
    M = length(ts)
    X = zeros(N,M)
    for (i,t) in enumerate(ts)
        X[:,i] = t == 0 ? X₀ : X[:,i-1] .+ rand(Δtξ,N) .- dUₓ.(X[:,i-1]).*Δt
    end
    return (ts=ts,X=X)
end

function distanim(ts::AbstractRange,X::Matrix{<:Real};F=10,xbins=:fd,ylim=(0,0.5))
    @gif for (f,(t,Xₜ)) in zip(ts,eachcol(X)) |> enumerate
        (f-1) % F != 0 && continue
        histogram(Xₜ,normalize=:pdf,bins=xbins,label=false)
        title!("Particles distribution at t=$t")
        ylims!(ylim)
        xlabel!("\$X(t=$t)\$")
        ylabel!("Probability Density")
    end
end

ts,X=randwalk(zeros(3000),500;σ²=2);

plotly()
plot(ts,X[end,:],label=false)
for i in 1:50
    plot!(ts,X[i,:],label=false)
end
title!("Particle trajectory")
xlabel!("t")
ylabel!("X")

gr();
distanim(0:0.1:150,X;F=10,xbins=-50:2:50,ylim=(0,0.2))

x̄²=mean(X.^2,dims=1)|>vec;

plotly()
plot(ts,x̄²,label=raw"$\bar{x^2}(numerical)$")
plot!(t->2t,label=raw"$2t(theoretical)$")
title!("Compare diffusivity: numerical and theoretical")
xlabel!(raw"$t$")
ylabel!(raw"$<[x(t)-x(0)]^2>$")

ts,X=randwalk(rand(Uniform(-5,5),5000),5000;σ²=2,dUₓ=identity);
Xₜ=X[:,end];

plotly()
histogram(Xₜ,normalize=:pdf,bins=-4:0.2:4,label="ePDF(t=$(ts[end]))")
plot!(fit(Normal,Xₜ),lw=3,label=" PDF(fit)")
xlabel!("\$X(t=$(ts[end]))\$")
ylabel!("Probability Density")

@show var(Xₜ);

gr();
distanim(0:0.1:12,X;F=1,xbins=-4:0.2:4,ylim=(0,0.5))


