using Oceananigans
using JLD2
using Printf, CairoMakie
using FilePathsBase: mkpath
using FilePathsBase

# Ensure OUTPUTS directory exists
mkpath(joinpath(@__DIR__, "OUTPUTS"))

@info "Set up model"

Nx, Ny = 64, 32
#Nx, Ny = 1024, 256
Lx = 2π
Ly = 20

grid = RectilinearGrid(size=(Nx, Ny),
                       x=(0, Lx),
                       y=(-Ly/2, Ly/2),
                       topology=(Periodic, Bounded, Flat)
)

gravitational_acceleration = 9.81

model = ShallowWaterModel(; grid,
                          gravitational_acceleration,
                          timestepper = :RungeKutta3)

@info "Set initial conditions"
uh, vh, h = model.solution
u = uh ./ h
v = vh ./ h

A₀, α₀, x₀, y₀, H = 1, 1, π, 0.5, 15

uᵢ(x, y) = A₀ * 2 * (y - y₀) * α₀ * exp(-α₀ * ((x - x₀)^2 + (y - y₀)^2)) -
           A₀ * 2 * (y + y₀) * α₀ * exp(-α₀ * ((x - x₀)^2 + (y + y₀)^2))

vᵢ(x, y) = -(A₀ * 2 * (x - x₀) * α₀ * exp(-α₀ * ((x - x₀)^2 + (y - y₀)^2)) -
             A₀ * 2 * (x - x₀) * α₀ * exp(-α₀ * ((x - x₀)^2 + (y + y₀)^2)))

h̄(x, y) = H

uhᵢ(x, y) = uᵢ(x, y) * h̄(x, y)
vhᵢ(x, y) = vᵢ(x, y) * h̄(x, y)

set!(model, uh = uhᵢ, vh = vhᵢ, h = h̄)

@info "Setting up fields"
ω = Field(∂x(v) - ∂y(u))
s = Field(sqrt(u^2 + v^2))

@info "Set up simulation"
simulation = Simulation(model, Δt = 1e-4, stop_time = 12)

@info "Set up progress message and timestep wizard"
wizard = TimeStepWizard(cfl = 0.7, max_change = 1.1, max_Δt = 1e-4)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(100))

function progress_message(sim)
    max_abs_u = maximum(abs, sim.model.velocities.u)
    max_abs_v = maximum(abs, sim.model.velocities.v)
    walltime = prettytime(sim.run_wall_time)
    @info @sprintf("Iteration: %04d, time: %1.3f, Δt: %.2e, max(|u|)=%.1e, max(|v|)=%.1e, wall: %s",
                   iteration(sim), time(sim), sim.Δt, max_abs_u, max_abs_v, walltime)
end

add_callback!(simulation, progress_message, IterationInterval(500))

@info "Set up output writers"
fields_filename  = joinpath(@__DIR__, "OUTPUTS", "dipole_shallow_water_fields.jld2")
heights_filename = joinpath(@__DIR__, "OUTPUTS", "dipole_shallow_water_heights.jld2")

simulation.output_writers[:fields] = JLD2OutputWriter(model, (ω, s);
    schedule = TimeInterval(0.5),
    filename = fields_filename,
    overwrite_existing = true
)

simulation.output_writers[:height] = JLD2OutputWriter(model, (h,);
    schedule = TimeInterval(0.5),
    filename = heights_filename,
    overwrite_existing = true
)

@info "Run the simulation"
run!(simulation)

@info "Load data from JLD2 files"
ω_ts = FieldTimeSeries(fields_filename, "ω")
s_ts = FieldTimeSeries(fields_filename, "s")
h_ts = FieldTimeSeries(heights_filename, "h")

times = ω_ts.times
println("Saved times: ", times)

# -----------------------------------------------------------------------------
# Wrap all plotting in a function to avoid soft‐scope warnings
# -----------------------------------------------------------------------------
function make_dipole_plots()
    x, y = xnodes(ω), ynodes(ω)

    # Precompute color ranges
    ωlims = (minimum(interior(ω_ts)), maximum(interior(ω_ts)))
    slims = (minimum(interior(s_ts)), maximum(interior(s_ts)))
    hlims = (minimum(interior(h_ts)), maximum(interior(h_ts)))

    # Set up figure
    fig = Figure(resolution = (1200, 1600), fontsize = 32)

    # Common axis kwargs
    fontsize = 28
    axis_kwargs = (
        xlabel = "x", ylabel = "y",
        xlabelsize = fontsize, ylabelsize = fontsize,
        xticklabelsize = fontsize, yticklabelsize = fontsize,
        xticks = (0:π/3:2π, ["0","π/3","2π/3","π","4π/3","5π/3","2π"]),
        yticks = -10:2:10,
        limits = ((0, 2π), (-10, 10)),
        titlefontsize = fontsize
    )

    # Lifted Observable index
    n = Observable(1)
    ω_field = @lift ω_ts[$n]
    s_field = @lift s_ts[$n]
    h_field = @lift h_ts[$n]

    # Helper to add a row of heatmap+colorbar
    function add_row(row, title, field, crange, cmap, unit_label)
        ax = Axis(fig[row, 1]; title = title, axis_kwargs...)
        hm = heatmap!(ax, x, y, field; colormap = cmap, colorrange = crange)
        Colorbar(fig[row, 2], hm;
                 label = unit_label,
                 labelsize = 20,
                 ticklabelsize = 20)
    end

    add_row(2, L"Vorticity $ω$",        ω_field, ωlims, :balance, L"$s^{-1}$")
    add_row(3, L"Velocity magnitude $|\mathbf v|$", s_field, slims, :speed, L"m/s")
    add_row(4, L"Height $h$",           h_field, hlims, :balance, L"m")

    # Time label at top
    fig[1, :] = Label(fig, @lift @sprintf("t = %.1f", times[$n]),
                      fontsize = 24, tellwidth = false)

    # Save and record animation
    save(joinpath(@__DIR__, "OUTPUTS", "dipole_shallow_water_total_vorticity.png"), fig)
    record(fig,
           joinpath(@__DIR__, "OUTPUTS", "dipole_shallow_water_total_vorticity_animation.mp4"),
           1:length(times); framerate = 8) do i
        n[] = i
    end

    # Static snapshots (4 panels)
    selected = round.(Int, range(1, stop = length(times), length = 4))
    snap_times = times[selected]

    fig2 = Figure(resolution = (3400, 1600))
    Label(fig2[1, 1:4], "Snapshots of the evolution of a dipole in shallow water", fontsize = 64)

    for (i, idx) in enumerate(selected)
        axω = Axis(fig2[2, i]; title = @sprintf("t = %.1f", snap_times[i]), titlefontsize = 42,
                   xlabel = "x", ylabel = "y",
                   xticks = (0:π/3:2π, []), yticks = -10:5:10,
                   limits = ((0,2π),(-10,10)),
                   xlabelsize = 42, ylabelsize = 42,
                   xticklabelsize = 32, yticklabelsize = 32)
        heatmap!(axω, x, y, ω_ts[idx]; colormap = :balance, colorrange = ωlims)

        axs = Axis(fig2[3, i]; xlabel = "x", ylabel = "y",
                   xticks = (0:π/3:2π, []), yticks = -10:5:10,
                   limits = ((0,2π),(-10,10)),
                   xlabelsize = 42, ylabelsize = 42,
                   xticklabelsize = 32, yticklabelsize = 32)
        heatmap!(axs, x, y, s_ts[idx]; colormap = :speed, colorrange = slims)

        axh = Axis(fig2[4, i]; xlabel = "x", ylabel = "y",
                   xticks = (0:π/3:2π, []), yticks = -10:5:10,
                   limits = ((0,2π),(-10,10)),
                   xlabelsize = 42, ylabelsize = 42,
                   xticklabelsize = 32, yticklabelsize = 32)
        heatmap!(axh, x, y, h_ts[idx]; colormap = :balance, colorrange = hlims)
    end

    Colorbar(fig2[2, 5]; colormap = :balance, colorrange = ωlims,
             ticklabelsize = 32, labelsize = 32, label = L"Vorticity $ω$")
    Colorbar(fig2[3, 5]; colormap = :speed,  colorrange = slims,
             ticklabelsize = 32, labelsize = 32, label = L"|v| (m/s)")
    Colorbar(fig2[4, 5]; colormap = :balance, colorrange = hlims,
             ticklabelsize = 32, labelsize = 32, label = L"Height (m)")

    save(joinpath(@__DIR__, "OUTPUTS", "dipole_shallow_water_snapshots.png"), fig2)
end

# Run the plotting function
make_dipole_plots()