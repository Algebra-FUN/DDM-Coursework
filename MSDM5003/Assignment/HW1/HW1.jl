# mute logging in jupyter notebook
import Logging
Logging.disable_logging(Logging.Warn)

# load dependency
using Distributions, Plots, StatsPlots

# change backend of Plots to Plotly
plotly();

# num of samples
n=1000;

# generate a seq of rand num Xs, X~N(0,1)
xs = rand(Normal(0,1),n)
# generate a seq of rand num Ys, Y~N(1,2)
ys = rand(Normal(1,√2),n)
# generate a seq of rand num Zs, Z=X+Y
zs = xs .+ ys

# implement ecdf-plot by my self.
# Pᶻs = Vector(1:n)./n
# zsₛₒᵣₜ = sort(zs)
# plot([repeat(zsₛₒᵣₜ',2)...],[[(Pᶻs.-1/n)';Pᶻs']...],legend=false)

ecdfplot(zs,legend=false)
title!("Empirical CDF")
xlabel!("Z")
ylabel!("Probability")

# calculate average of samples: zs
μ_=mean(zs);
# calculate variance of samples: zs
σ²_=var(zs);
print("̂μ = $μ_ ,̂σ² = $σ²_")

z_range = -4:.01:6
ecdfplot(zs,label="eCDF")
plot!(z->cdf(Normal(μ_,√σ²_),z),z_range,label="  CDF")
title!("Cumulative Distribution")
xlabel!("Z")
ylabel!("Probability")

d = fit(Normal,zs)

histogram(zs,normalize=:pdf,label="ePDF")
plot!(fit(Normal,zs),lw=3,color=:red,label=" PDF")
title!("Density")
xlabel!("Z")
ylabel!("Probability Density")


