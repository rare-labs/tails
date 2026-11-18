# Reference values from IDFCurves.jl (upstream commit 66a34d1) for validating tails.idf.
# Usage: julia --project=validation/julia validation/idfcurves_reference.jl > tests/data/idfcurves_reference.json
# The Manifest pins IDFCurves to a local checkout of the upstream repository at that commit
# (edit the `path` entry, or `Pkg.add(url="https://github.com/JuliaExtremes/IDFCurves.jl", rev="66a34d1")`).
using IDFCurves, DataFrames, Distributions, LinearAlgebra

fmt(x::Real) = isfinite(x) ? repr(Float64(x)) : "null"
fmt(v::AbstractVector) = "[" * join(fmt.(v), ",") * "]"

durs = [1/12, 1/6, 1/4, 1/2, 1, 2, 6, 12, 24]
models = [
    ("simple", SimpleScaling),
    ("general", GeneralScaling),
    ("general_matern_gaussian", DependentScalingModel{GeneralScaling, MaternCorrelationStructure, GaussianCopula}),
    ("general_exponential_gaussian", DependentScalingModel{GeneralScaling, ExponentialCorrelationStructure, GaussianCopula}),
    ("general_exponential_t5", DependentScalingModel{GeneralScaling, ExponentialCorrelationStructure, TCopula{5}}),
]
out = String[]
for st in ["702S006", "6158731", "1108446"]
    df = dropmissing(IDFCurves.dataset(st))
    tags = names(df)[2:10]
    data = IDFdata(df, "Year", Dict(zip(tags, durs)))
    entries = String[]
    for (name, T) in models
        t = @elapsed m = IDFCurves.fit_mle(T, data, 1.0)
        θ = collect(params(m))
        ll = loglikelihood(m, data)
        fields = ["\"params\": " * fmt(θ), "\"loglik\": " * fmt(ll), "\"fit_seconds_incl_compile\": " * fmt(t)]
        for (d, lab) in ((1.0, "1h"), (24.0, "24h"), (1/12, "5min"))
            push!(fields, "\"q100_$lab\": " * fmt(quantile(m, d, 0.99)))
            try
                push!(fields, "\"ci100_$lab\": " * fmt(quantilecint(m, data, d, 0.99)))
            catch
            end
        end
        if name == "general"
            G = IDFCurves.godambe(m, data)
            push!(fields, "\"ci100_1h_godambe\": " * fmt(quantilecint(m, data, 1.0, 0.99, G)))
            push!(fields, "\"scalingtest\": " * fmt(IDFCurves.scalingtest(GeneralScaling, data)))
        elseif name == "simple"
            push!(fields, "\"scalingtest\": " * fmt(IDFCurves.scalingtest(SimpleScaling, data)))
        end
        push!(entries, "\"$name\": {" * join(fields, ", ") * "}")
    end
    push!(out, "\"$st\": {\"n_years\": $(nrow(df)), " * join(entries, ", ") * "}")
end
println("{" * join(out, ",\n") * "}")
