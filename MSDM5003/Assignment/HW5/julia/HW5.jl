# mute logging in jupyter notebook
import Logging
Logging.disable_logging(Logging.Warn)

# load dependency
using DataFrames, Statistics, Plots, GLM
import CSV

df = CSV.read("^HSI.csv",DataFrame)

df = df[.!isnan.(df.Close),:]

prices = df.Close;

r = log.(prices[2:end]./prices[1:end-1]);

function normalize(x)
    σ=std(x)
    μ=mean(x)
    return (x.-μ)./σ
end

r̃ = normalize(r)

r⁺ = r[r .> 0];
r⁻ = r[r .< 0];

function ecdf(rs::Vector{<:Real})
    return x->mean(rs .> x)
end

function power_law_fit(rs::Vector{<:Real},xs)
    y=ecdf(rs).(xs)
    data=DataFrame(X=log10.(xs[y.>0]),Y=log10.(y[y.>0]))
    model=lm(@formula(Y ~ X), data)
    a,b=coef(model)
    return (α=-b,c=a,R²=r²(model))
end

xs=.01:.001:.2

α,c,R²=power_law_fit(r⁺,xs)
plotly()
scatter(xs,ecdf(r⁺).(xs),label="observation")
plot!(x->(10^c)*(x^(-α)),xs,label="fit(α=$(round(α;digits=3)))")
xlabel!("log(return⁺)")
ylabel!("log(cumulative probability)")
title!("Postive Tail")
annotate!(-2.5,-2,text("y=$(round(10^c;digits=6))x^{-$(round(α;digits=3))}",:left))
annotate!(-2.5,-2.5,text("R²=$(round(R²;digits=4))",:left))
xaxis!(:log)
yaxis!(:log)
ylims!(10^-4,10^0)
xlims!(10^-2.5,10^0)

absr⁻=abs.(r⁻)
α,c,R²=power_law_fit(absr⁻,xs)
plotly()
scatter(xs,ecdf(absr⁻).(xs),label="observation")
plot!(x->(10^c)*(x^(-α)),xs,label="fit(α=$(round(α;digits=3)))")
xlabel!("log(abs(return⁻))")
ylabel!("log(cumulative probability)")
title!("Negative Tail")
annotate!(-2.5,-2,text("y=$(round(10^c;digits=6))x^{-$(round(α;digits=3))}",:left))
annotate!(-2.5,-2.5,text("R²=$(round(R²;digits=4))",:left))
xaxis!(:log)
yaxis!(:log)
ylims!(10^-4,10^0)
xlims!(10^-2.5,10^0)
