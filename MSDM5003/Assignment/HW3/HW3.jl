function Simpson(f::Function,interval::Tuple{<:Real,<:Real};n::Integer=100)
    a,b = interval
    h = (b-a)/n
    fx(k) = f(a+k*h)
    Sₙ = h/6*sum((fx(k)+4*fx(k+.5)+fx(k+1)) for k in 0:n-1)
    return Sₙ
end;

s₀(x) = -3<=x<=3 ? sin(2π*x) : 0

p(x) = -3<=x<=3 ? sin(20π*x) : 0

s(x) = s₀(x) + p(x);

function correlation(s₁::Function,s₂::Function,interval::Tuple{<:Real,<:Real};n=100)
    f(x,y) = s₁(y)*s₂(x+y)
    return x->Simpson(y->f(x,y),interval;n=n)
end;

f(x) = correlation(s₀,s₀,(-10,10);n=100)(x);

# mute logging in jupyter notebook
import Logging
Logging.disable_logging(Logging.Warn)

# load dependency
using Distributions, Plots, Printf

plotly();

xs = -9:0.01:9
plotly()
plot(f,xs,label="cross correlation of s₀ and s₀")
plot!(s₀,xs,label="signal s₀")
xlabel!("x")

r(x;σ=0.2)=pdf(Normal(0,σ),x);

function convolution(s::Function,r::Function,interval::Tuple{<:Real,<:Real};n=100)
    f(x,y) = s(y)*r(x-y)
    return x->Simpson(y->f(x,y),interval;n=n)
end;

s₁(x) = convolution(s,r,(-6,6);n=1000)(x);

xs = -6:0.01:6
plotly()
plot(s,xs,label="origin signal s")
plot!(s₁,xs,label="new signal s₁ convoluted from s",lw=3)
xlabel!("x")

plotly()
xs = -6:0.01:6
len = length(xs)
plot(zeros(len),xs,s.(xs),label="signal")

for σ in 0.05:0.05:0.5
    s₁(x) = convolution(s,x->r(x;σ=σ),(-6,6);n=1000)(x)
    σs = σ.*ones(len)
    plot!(σs,xs,s₁.(xs),label="σ=$σ",lw=3)
end

ylabel!("x")
xlabel!("σ")

gr();
xs = -6:0.01:6
@gif for σ in 0.01:0.01:0.5
    s₁(x) = convolution(s,x->r(x;σ=σ),(-6,6);n=1000)(x)
    plot(s,xs,label="signal s")
    plot!(s₁,xs,label="convolution of s with r",lw=2)
    xlabel!("x")
    title!("convolution with r(x;σ=$(@sprintf("%1.2f",σ)))")
end
