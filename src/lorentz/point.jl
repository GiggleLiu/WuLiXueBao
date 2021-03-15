struct P3{T}
    x::T
    y::T
    z::T
end

Base.zero(::Type{P3{T}}) where T = P3(zero(T), zero(T), zero(T))
Base.zero(::P3{T}) where T = P3(zero(T), zero(T), zero(T))
