# Generalized cross-validation for smoothing splines
using Optim

export GCV

struct GCV
    a
    b
end

score(res::AbstractVector,resmat::AbstractMatrix,n=length(res)) = n*res'res/trace(resmat)^2

score(res::AbstractVector,restrace,n=length(res)) = n*res'res/restrace^2

function score(spl::SmoothingSpline)
    res = residuals(spl)
    restrace = residualtrace(spl)
    score(res,restrace)
end

score(λ,X::AbstractVector,Y::AbstractVector,wts::AbstractVector) = score(fit(SmoothingSpline,X,Y,λ,wts))

function residualtrace(spl::SmoothingSpline)
    h = diff(spl.Xdesign)
    Q = ReinschQ(h)
    Qt = Q'
    λ = spl.λ
    RpαQtQ = spl.RpαQtQ
    
    pbtrs!('U',2,spl.RpαQtQ,Qt)
    g = trace_multiply!(zeros(size(Q,1)),Q,Qt)
    broadcast!(/,g,g,spl.weights)
    broadcast!(*,g,g,λ)
    sum(g)
end

function gcv(X::AbstractVector,Y::AbstractVector,wts::AbstractVector,opts::GCV)
    res = optimize(λ->score(λ,X,Y,wts),opts.a,opts.b)
    fit(SmoothingSpline,X,Y,Optim.minimizer(res))
end
    
