




############################################################################
#                          Draw the Decision Set                           #
############################################################################



function mklocsvmplot(data::AbstractMatrix{Float64}; kwargs...)

    n = size(data, 1)

    if n == 2
        fig, ax, _, _, backend = _myplotformat(data; kwargs...)
        backend.scatter!(ax, data[1, :], data[2, :])
    elseif n == 3
        fig, ax, _, _, _, backend = _myplotformat3d(data; kwargs...)
        backend.scatter!(ax, data[1, :], data[2, :], data[3, :])
    else
        @error "Unsupported data dimension."
    end

    display(fig)

    return fig, ax
end

function mklocsvmplot(
    model::StandardModel;
    xslims=nothing,
    yslims=nothing,
    zslims=nothing,
    xslen=nothing,
    yslen=nothing,
    zslen=nothing,
    contour_color=:red,
    contour_linewidth=2.0,
    contourf_color=(:red, 0.2),
    iv_color=(:royalblue1, 0.75),
    bsv_color=:blue,
    osv_color=:purple,
    contour3d_colormap=:reds, # https://docs.makie.org/stable/explanations/colors/
    contour3d_alpha=0.1,
    kwargs...
)

    data = model.data
    data_sv = model.data_sv
    data_bsv = model.data_bsv
    data_osv = model.data_osv
    data_iv = model.data_iv
    kernels_sk = model.kernels_sk
    α_sv = model.α_sv
    π_sk = model.π_sk
    ρ = model.ρ
    nsv = length(α_sv)
    nsk = length(π_sk)
    ϵ = model.ϵ

    n = size(data, 1)

    if n == 2
        fig, ax, xlims, ylims, backend = _myplotformat(data; kwargs...)

        if isnothing(xslims)
            xslims = xlims
        end
        if isnothing(yslims)
            yslims = ylims
        end
        if isnothing(xslen)
            xslen = 200
        end
        if isnothing(yslen)
            yslen = 200
        end
        xs = LinRange(xslims..., xslen)
        ys = LinRange(yslims..., yslen)
        zs = [sum(α_sv[i] * π_sk[m] * kernels_sk[m]([x, y], data_sv[:,i]) for i in 1:nsv, m in 1:nsk) - ρ for x in xs, y in ys]
    
        backend.contourf!(ax, xs, ys, zs; levels=[-ϵ, maximum(zs[:])], colormap=[contourf_color])
        backend.scatter!(ax, data_iv[1, :], data_iv[2, :]; color=iv_color)
        backend.contour!(ax, xs, ys, zs; levels=[-ϵ], linewidth=contour_linewidth, color=contour_color)
        backend.scatter!(ax, data_bsv[1, :], data_bsv[2, :]; color=bsv_color)
        backend.scatter!(ax, data_osv[1, :], data_osv[2, :]; color=osv_color)
    elseif n == 3
        fig, ax, xlims, ylims, zlims, backend = _myplotformat3d(data; kwargs...)

        if isnothing(xslims)
            xslims = xlims
        end
        if isnothing(yslims)
            yslims = ylims
        end
        if isnothing(zslims)
            zslims = zlims
        end
        if isnothing(xslen)
            xslen = 10
        end
        if isnothing(yslen)
            yslen = 10
        end
        if isnothing(zslen)
            zslen = 10
        end
        xs = LinRange(xslims..., xslen)
        ys = LinRange(yslims..., yslen)
        zs = LinRange(zslims..., zslen)
        torus = [sum(α_sv[i] * π_sk[m] * kernels_sk[m]([x, y, z], data_sv[:,i]) for i in 1:nsv, m in 1:nsk) - ρ for x in xs, y in ys, z in zs]

        backend.scatter!(ax, data_iv[1, :], data_iv[2, :], data_iv[3, :]; color=iv_color)
        backend.contour!(ax, (xs[1], xs[end]), (ys[1], ys[end]), (zs[1], zs[end]), torus; levels=[-ϵ], colormap=contour3d_colormap, alpha=contour3d_alpha)
        backend.scatter!(ax, data_bsv[1, :], data_bsv[2, :], data_bsv[3, :]; color=bsv_color)
        backend.scatter!(ax, data_osv[1, :], data_osv[2, :], data_osv[3, :]; color=osv_color)
    else
        @error "Unsupported data dimension."
    end

    display(fig)

    return fig, ax
end

function mklocsvmplot(
    model::Vector{StandardModel};
    xslims=nothing,
    yslims=nothing,
    zslims=nothing,
    xslen=nothing,
    yslen=nothing,
    zslen=nothing,
    contour_color=:red,
    contour_linewidth=2.0,
    contourf_color=(:red, 0.2),
    iv_color=(:royalblue1, 0.75),
    bsv_color=:blue,
    osv_color=:purple,
    contour3d_colormap=:reds, # https://docs.makie.org/stable/explanations/colors/
    contour3d_alpha=0.1,
    kwargs...
)

    data = model[1].data
    n, N = size(data)
    num_batch = length(model)
    ϵ₀ = mean([model[bat].ϵ for bat in 1:num_batch])

    if n == 2
        fig, ax, xlims, ylims, backend = _myplotformat(data; kwargs...)

        if isnothing(xslims)
            xslims = xlims
        end
        if isnothing(yslims)
            yslims = ylims
        end
        if isnothing(xslen)
            xslen = 200
        end
        if isnothing(yslen)
            yslen = 200
        end
        xs = LinRange(xslims..., xslen)
        ys = LinRange(yslims..., yslen)
        intsct = [true for x in xs, y in ys]
        zsprod = [1. for x in xs, y in ys]
    
        OSV = Int64[]
        IV = [1:N;]
        for bat in 1:num_batch
            OSV = OSV ∪ model[bat].OSV
            IV = IV ∩ model[bat].IV
        end
        BSV = setdiff([1:N...], OSV, IV)
        for bat in 1:num_batch
            data_sv = model[bat].data_sv
            kernels_sk = model[bat].kernels_sk
            α_sv = model[bat].α_sv
            π_sk = model[bat].π_sk
            ρ = model[bat].ρ
            nsv = length(α_sv)
            nsk = length(π_sk)
    
            zs = [sum(α_sv[i] * π_sk[m] * kernels_sk[m]([x, y], data_sv[:,i]) for i in 1:nsv, m in 1:nsk) - ρ for x in xs, y in ys]
    
            intsct = intsct .& (zs .>= -model[bat].ϵ)
            zsprod[intsct] = zsprod[intsct] .* abs.(zs[intsct])
            zsprod[.~intsct] = zsprod[.~intsct] + abs.(zs[.~intsct])
    
            backend.contour!(ax, xs, ys, zs; levels = [-model[bat].ϵ], color=contour_color)
        end
        zsprod[.~intsct] = -zsprod[.~intsct]
        backend.contourf!(ax, xs, ys, zsprod; levels=[-ϵ₀, maximum(zsprod[:])], colormap=[contourf_color])
        backend.scatter!(ax, data[1, IV], data[2, IV]; color=iv_color)
        backend.contour!(ax, xs, ys, zsprod; levels = [-ϵ₀], linewidth=contour_linewidth, color=contour_color)
        backend.scatter!(ax, data[1, BSV], data[2, BSV]; color=bsv_color)
        backend.scatter!(ax, data[1, OSV], data[2, OSV]; color=osv_color)
    elseif n == 3
        fig, ax, xlims, ylims, zlims, backend = _myplotformat3d(data; kwargs...)

        if isnothing(xslims)
            xslims = xlims
        end
        if isnothing(yslims)
            yslims = ylims
        end
        if isnothing(zslims)
            zslims = zlims
        end
        if isnothing(xslen)
            xslen = 10
        end
        if isnothing(yslen)
            yslen = 10
        end
        if isnothing(zslen)
            zslen = 10
        end
        xs = LinRange(xslims..., xslen)
        ys = LinRange(yslims..., yslen)
        zs = LinRange(zslims..., zslen)
        intsct = [true for x in xs, y in ys, z in zs]
        toursprod = [1. for x in xs, y in ys, z in zs]

        OSV = Int64[]
        IV = [1:N;]
        for bat in 1:num_batch
            OSV = OSV ∪ model[bat].OSV
            IV = IV ∩ model[bat].IV
        end
        BSV = setdiff([1:N...], OSV, IV)
        for bat in 1:num_batch
            data_sv = model[bat].data_sv
            kernels_sk = model[bat].kernels_sk
            α_sv = model[bat].α_sv
            π_sk = model[bat].π_sk
            ρ = model[bat].ρ
            nsv = length(α_sv)
            nsk = length(π_sk)
    
            tours = [sum(α_sv[i] * π_sk[m] * kernels_sk[m]([x, y, z], data_sv[:,i]) for i in 1:nsv, m in 1:nsk) - ρ for x in xs, y in ys, z in zs]
    
            intsct = intsct .& (tours .>= -model[bat].ϵ)
            toursprod[intsct] = toursprod[intsct] .* abs.(tours[intsct])
            toursprod[.~intsct] = toursprod[.~intsct] + abs.(tours[.~intsct])
    
        end
        toursprod = (toursprod .* intsct) + (-rand(size(intsct))) * (.~intsct)
        toursprod[.~intsct] = -toursprod[.~intsct]
 
        backend.scatter!(ax, data[1, IV], data[2, IV], data[3, IV]; color=iv_color)
        backend.contour!(ax, (xs[1], xs[end]), (ys[1], ys[end]), (zs[1], zs[end]), toursprod; levels = [-ϵ₀, ϵ₀], colormap=contour3d_colormap, alpha=contour3d_alpha)
        backend.scatter!(ax, data[1, BSV], data[2, BSV], data[3, BSV]; color=bsv_color)
        backend.scatter!(ax, data[1, OSV], data[2, OSV], data[3, OSV]; color=osv_color)
    else
        @error "Unsupported data dimension."
    end

    display(fig)

    return fig, ax
end

function mklocsvmplot(model::CombinedModel; kwargs...)
    model = mklocsvmunbuild(model)
    mklocsvmplot(model; kwargs...)
end


############################################################################
#                                Utilities                                 #
############################################################################

function _myplotformat(
    data::AbstractMatrix{Float64}; 
    title="",
    xlabel="",
    ylabel="",
    xlims=nothing,
    ylims=nothing,
    aspect=nothing,
    margin=1/5,
    zoom=[1, 1],
    offset=[0, 0],
    backend::Module
)

    backend.activate!()

    if isnothing(xlims) || isnothing(ylims)
        xmax, ymax = maximum(data; dims=2)[1], maximum(data; dims=2)[2]
        xmin, ymin = minimum(data; dims=2)[1], minimum(data; dims=2)[2]
        w, h = xmax - xmin, ymax - ymin
        if isnothing(xlims)
            xlmin = xmin - margin * w
            xlmax = xmax + margin * w
            center = (xlmax + xlmin) / 2
            Δx = xlmax - xlmin
            xlmin = center - zoom[1] * Δx / 2
            xlmax = center + zoom[1] * Δx / 2
            xlims = [xlmin, xlmax] .+ offset[1]
        end
        if isnothing(ylims)
            ylmin = ymin - margin * h
            ylmax = ymax + margin * h
            center = (ylmax + ylmin) / 2
            Δy = ylmax - ylmin
            ylmin = center - zoom[2] * Δy / 2
            ylmax = center + zoom[2] * Δy / 2
            ylims = [ylmin, ylmax] .+ offset[2]
        end
    end
    if isnothing(aspect)
        aspect = (xlims[2] - xlims[1]) / (ylims[2] - ylims[1])
    end
    
    fig = backend.Figure()
    ax = backend.Axis(
        fig[1,1];
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
        aspect = aspect
    )
    backend.xlims!(ax, xlims)
    backend.ylims!(ax, ylims)

    return fig, ax, xlims, ylims, backend
end


function _myplotformat3d(
    data::AbstractMatrix{Float64}; 
    title="",
    xlabel="x",
    ylabel="y",
    zlabel="z",
    xlims=nothing,
    ylims=nothing,
    zlims=nothing,
    aspect=nothing,
    margin=1/5,
    zoom=[1, 1, 1],
    offset=[0, 0, 0],
    azimuth=1.275π,
    elevation=π/8,
    perspectiveness=0,
    backend::Module
)

    backend.activate!()

    if isnothing(xlims) || isnothing(ylims) || isnothing(zlims)
        xmax, ymax, zmax = maximum(data; dims=2)[1], maximum(data; dims=2)[2], maximum(data; dims=2)[3]
        xmin, ymin, zmin = minimum(data; dims=2)[1], minimum(data; dims=2)[2], minimum(data; dims=2)[3]
        l, w, h = xmax - xmin, ymax - ymin, zmax - zmin
        if isnothing(xlims)
            xlmin = xmin - margin * l
            xlmax = xmax + margin * l
            center = (xlmax + xlmin) / 2
            Δx = xlmax - xlmin
            xlmin = center - zoom[1] * Δx / 2
            xlmax = center + zoom[1] * Δx / 2
            xlims = [xlmin, xlmax] .+ offset[1]
        end
        if isnothing(ylims)
            ylmin = ymin - margin * w
            ylmax = ymax + margin * w
            center = (ylmax + ylmin) / 2
            Δy = ylmax - ylmin
            ylmin = center - zoom[2] * Δy / 2
            ylmax = center + zoom[2] * Δy / 2
            ylims = [ylmin, ylmax] .+ offset[2]
        end
        if isnothing(zlims)
            zlmin = zmin - margin * h
            zlmax = zmax + margin * h
            center = (zlmax + zlmin) / 2
            Δz = zlmax - zlmin
            zlmin = center - zoom[3] * Δz / 2
            zlmax = center + zoom[3] * Δz / 2
            zlims = [zlmin, zlmax] .+ offset[3]
        end
    end
    if isnothing(aspect)
        aspect = ((xlims[2] - xlims[1]), (ylims[2] - ylims[1]), (zlims[2] - zlims[1]))
    end

    fig = backend.Figure()
    ax = backend.Axis3(
        fig[1,1];
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
        zlabel = zlabel,
        limits= (xlims, ylims, zlims),
        aspect = aspect,
        azimuth=azimuth,
        elevation=elevation,
        perspectiveness=perspectiveness
    )

    return fig, ax, xlims, ylims, zlims, backend
end
