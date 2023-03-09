using LambdaFn
import Markdown

eye(n) = Matrix{Float64}(I,n,n)

function matrix2latex(M)
    rows_str = @λ(join(M[_,:],'&')) |> @λ map(_,1:size(M)[1])
    matrix_str = join(rows_str,"\\\\")
    return "\\begin{pmatrix}$matrix_str\\end{pmatrix}"
end

centralize(content::String) = "<center>``$content``</center>"

matrix_form(name::String,M) = ""*name*"="*matrix2latex(M)*""

matrix_form(symbol::Symbol,M) = matrix_form(String(symbol),M)

macro latex(M::Symbol)
    return matrix_form(M,M |> eval) |> centralize |> Markdown.parse
end

macro latex(M::Symbol,T::Symbol)
    if T == :T
        return matrix_form(M,(M |> eval)')*raw"^\top" |> centralize |> Markdown.parse
    end
    return matrix_form(M,M |> eval) |> centralize |> Markdown.parse
end

macro latex(expr::Expr)
    if expr.head == :(=)
        left = expr.args[1]
        right = expr.args[2] |> eval
        if typeof(left) == Symbol && any(typeof(right) .<: (Matrix,Vector))
            expr |> eval
            return matrix_form(left,right) |> centralize |> Markdown.parse
        end
    end
    return expr
end