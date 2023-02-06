# mute logging in jupyter notebook
import Logging
Logging.disable_logging(Logging.Warn)

# load dependency
using Distributions, Plots, StatsPlots, HypothesisTests

function Metorpolis(f::Function;x₀=0,g=x->Normal(x,1),n=10000)
    x=zeros(n)
    x[1] = x₀
    for t in 1:n-1
        xᶜ = rand(g(x[t])) 
        α = f(xᶜ)/f(x[t])
        u = rand()
        x[t+1] = u <= α ? xᶜ : x[t]
    end
    return x
end

f(x) = exp(-x^2/2)

X = Metorpolis(f)

for k in 1:4
    EXᵏ=mean(X.^k)
    println("E[x^$k]=$EXᵏ")
end

plotly()
histogram(X,normalize=:pdf,bins=-4:0.1:4,label="sequence x")
plot!(fit(Normal,X),label="PDF(fit)",lw=3,linecolor=:green)
plot!(Normal(0,1),label="PDF(theory)",lw=3,linecolor=:red)
xlabel!("X")

fit(Normal,X)

ExactOneSampleKSTest(X,Normal(0,1))
