# Generalized cross-validation for smoothing splines
export GCV

GCV(res::AbstractVector,resmat::AbstractMatrix,n=length(res)) = n*res'res/trace(resmat)^2

function GCV(spl::SmoothingSpline)
    res = residuals(spl)
    resmat = residualmatrix(spl)
    GCV(res,resmat)
end

GCV(λ,X::AbstractVector,Y::AbstractVector) = GCV(fit(SmoothingSpline,X,Y,λ))

function residualmatrix(spl::SmoothingSpline)
    h = diff(spl.Xdesign)
    Q = ReinschQ(h)
    Qt = Q'
    λ = spl.λ
    RpαQtQ = spl.RpαQtQ
    
    pbtrs!('U',2,spl.RpαQtQ,Qt)
    g = Q*Qt
    broadcast!(/,g,g,spl.weights)
    broadcast!(*,g,g,λ)
    g
end
