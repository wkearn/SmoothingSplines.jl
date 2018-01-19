# Generalized cross-validation for smoothing splines
export GCV

GCV(res::AbstractVector,resmat::AbstractMatrix,n=length(res)) = n*res'res/trace(resmat)^2

GCV(res::AbstractVector,restrace,n=length(res)) = n*res'res/restrace^2

function GCV(spl::SmoothingSpline)
    res = residuals(spl)
    resmat = residualtrace(spl)
    GCV(res,resmat)
end

GCV(λ,X::AbstractVector,Y::AbstractVector) = GCV(fit(SmoothingSpline,X,Y,λ))

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
