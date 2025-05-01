#!/usr/bin/env julia

using Oceananigans
using JLD2
using Printf, CairoMakie

# ---------------------------------------------------------------------------
# 0) Ensure OUTPUTS directory exists
# ---------------------------------------------------------------------------
mkpath(joinpath(@__DIR__, "OUTPUTS"))

# ---------------------------------------------------------------------------
# 1) Build the shallow-water model
# ---------------------------------------------------------------------------
@info "Set up model"
Nx, Ny = 64, 32           # test resolution; bump to 1024×256 later
Lx, Ly = 2π, 20.0

grid = RectilinearGrid(
  size     = (Nx, Ny),
  x        = (0, Lx),
  y        = (-Ly/2, Ly/2),
  topology = (Periodic, Bounded, Flat),
)

g = 9.81
model = ShallowWaterModel(
  grid                      = grid,
  gravitational_acceleration = g,
  timestepper               = :RungeKutta3,
)

# ---------------------------------------------------------------------------
# 1b) Unpack the prognostic fields so `h` exists
# ---------------------------------------------------------------------------
uh, vh, h = model.solution

# ---------------------------------------------------------------------------
# 2) Define your explicit dipole initial conditions
# ---------------------------------------------------------------------------
@info "Set initial conditions"

A₀, α₀, x₀, y₀, H = 1.0, 1.0, π, 0.5, 15.0

uᵢ(x, y) = A₀*2*(y - y₀)*α₀*exp(-α₀*((x - x₀)^2 + (y - y₀)^2)) -
           A₀*2*(y + y₀)*α₀*exp(-α₀*((x - x₀)^2 + (y + y₀)^2))

vᵢ(x, y) = -(A₀*2*(x - x₀)*α₀*exp(-α₀*((x - x₀)^2 + (y - y₀)^2)) -
            A₀*2*(x - x₀)*α₀*exp(-α₀*((x - x₀)^2 + (y + y₀)^2)))

h̄(x, y) = H
uhᵢ(x, y) = uᵢ(x, y)*h̄(x, y)
vhᵢ(x, y) = vᵢ(x, y)*h̄(x, y)

set!(model, uh = uhᵢ, vh = vhᵢ, h = h̄)

# ---------------------------------------------------------------------------
# 3) Define diagnostics
# ---------------------------------------------------------------------------
@info "Setting up fields"
u = model.velocities.u
v = model.velocities.v

ω = Field(∂x(v) - ∂y(u))    # vorticity
s = Field(sqrt(u^2 + v^2))  # speed

# ---------------------------------------------------------------------------
# 4) Simulation setup + run
# ---------------------------------------------------------------------------
@info "Set up simulation"
simulation = Simulation(model, Δt=1e-4, stop_time=12)

wizard = TimeStepWizard(cfl=0.7, max_change=1.1, max_Δt=1e-4)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(500))

function progress_message(sim)
    max_u = maximum(abs, sim.model.velocities.u)
    max_v = maximum(abs, sim.model.velocities.v)
    wt    = prettytime(sim.run_wall_time)
    @info @sprintf("Iter %06d  t=%.3f  Δt=%.2e  max|u|=%.2e  max|v|=%.2e  wall=%s",
                   iteration(sim), time(sim), sim.Δt, max_u, max_v, wt)
end

add_callback!(simulation, progress_message, IterationInterval(500))

fields_file  = joinpath(@__DIR__, "OUTPUTS", "dipole_fields.jld2")
height_file  = joinpath(@__DIR__, "OUTPUTS", "dipole_height.jld2")

simulation.output_writers[:fields] = JLD2OutputWriter(
  model, (; ω, s);
  schedule           = TimeInterval(0.5),
  filename           = fields_file,
  overwrite_existing = true,
)

# ← here we now use a *named* tuple (; h) so that it writes the field "h"
simulation.output_writers[:height] = JLD2OutputWriter(
  model, (; h);
  schedule           = TimeInterval(0.5),
  filename           = height_file,
  overwrite_existing = true,
)

@info "Run the simulation"
run!(simulation)

# ---------------------------------------------------------------------------
# 5) Load results and plot + animate
# ---------------------------------------------------------------------------
@info "Load data from JLD2 files"
ω_ts = FieldTimeSeries(fields_file, "ω")
s_ts = FieldTimeSeries(fields_file, "s")
h_ts = FieldTimeSeries(height_file, "h")
times = ω_ts.times
println("Saved times: ", times)

function make_plots()
    x = Array(xnodes(grid, Center()))
    y = Array(ynodes(grid, Center()))

    # color ranges
    ωlims = extrema(interior(ω_ts))
    slims = extrema(interior(s_ts))
    hlims = extrema(interior(h_ts))

    fig = Figure(resolution=(1200,1600), fontsize=32)

    # tightened font sizes and y-limits ±5
    axis_fs   = 26
    tick_fs   = 22
    title_fs  = 30
    cbar_fs   = 20

    common_kwargs = (
      xlabel        = "x",
      ylabel        = "y",
      xlabelsize    = axis_fs,
      ylabelsize    = axis_fs,
      xticklabelsize= tick_fs,
      yticklabelsize= tick_fs,
      xticks        = (0:π/3:2π, ["0","π/3","2π/3","π","4π/3","5π/3","2π"]),
      yticks        = -5:1:5,
      limits        = ((0,2π),(-5,5)),
      titlefontsize = title_fs,
    )

    n = Observable(1)
    ω_field = @lift ω_ts[$n]
    s_field = @lift s_ts[$n]
    h_field = @lift h_ts[$n]

    function add_row(row, title, field, crange, cmap, cbar_label)
      ax = Axis(fig[row,1];
                title = title,
                common_kwargs...,
                titleposition = :top)
      hm = heatmap!(ax, x, y, field;
                    colormap   = cmap,
                    colorrange = crange)
      Colorbar(fig[row,2], hm;
               label         = cbar_label,
               labelsize     = cbar_fs,
               ticklabelsize = tick_fs)
    end

    add_row(2, L"Vorticity, $ω$",       ω_field, ωlims, :balance, L"Vorticity [s⁻¹]")
    add_row(3, L"Speed, $|\mathbf v|$",  s_field, slims, :speed,   L"|v| [m/s]")
    add_row(4, L"Height, $h$",          h_field, hlims, :balance, L"Height [m]")

    fig[1,:] = Label(fig, @lift @sprintf("t = %.1f", times[$n]);
                     fontsize = 28, tellwidth = false)

    save(joinpath(@__DIR__, "OUTPUTS", "dipole_vorticity.png"), fig)
    record(fig,
           joinpath(@__DIR__, "OUTPUTS", "dipole_vorticity.mp4"),
           1:length(times); framerate=8) do i
      n[] = i
    end
end

make_plots()