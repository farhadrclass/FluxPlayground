using flux


cfun(x::AbstractArray) = Float64.(x); 
cfun(x) = x; 

# m64 = Flux.fmap(cfun, predict);

f16(m) = Flux.paramtype(Float16, m) # similar to https://github.com/FluxML/Flux.jl/blob/d21460060e055dca1837c488005f6b1a8e87fa1b/src/functor.jl#L217
m16 = f16(m64)
