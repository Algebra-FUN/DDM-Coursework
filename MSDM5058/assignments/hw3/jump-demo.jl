using JuMP
using Ipopt
using GLPK

A = [
    3 5 3;
    4 -3 2;
    3 2 3
]

model = Model(Ipopt.Optimizer)
@variable(model,z)
@variable(model,x[1:3])
@constraint(model, sum(x)==1)
@constraint(model, x .>= 0)
@constraint(model,z .<= A'*x)
@objective(model,Max,z)
optimize!(model)
value.(x)

sum(value.(x))

value(z)

A'*value.(x)

model = Model(GLPK.Optimizer)
@variable(model,x[1:3])
@constraint(model, x .>= 0)
@constraint(model, A'*x .>= 1)
@objective(model,Min,sum(x))
optimize!(model)
value.(x)

x = value.(x)
