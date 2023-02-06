f(x)=2cos(2x)+4x*sin(x)-x^2-2
df(x)=-4sin(2x)+4sin(x)+4x*cos(x)-2x
x = π/2
for k in 1:4
    Δx = -f(x)/df(x)
    xꜝ = x + Δx
    println("\$k=$k,x^{($k)}=x^{($(k-1))}-\\frac{2\\cos(2x^{($(k-1))})+4x\\sin(x^{($(k-1))})-{x^{($(k-1))}}^2-2}{-4\\sin(2x^{($(k-1))})+4\\sin(x^{($(k-1))})+4x\\cos(x^{($(k-1))})-2x^{($(k-1))}}=$xꜝ\$")
    global x = xꜝ
end