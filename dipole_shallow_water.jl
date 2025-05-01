#!/usr/bin/env julia

using Oceananigans
using JLD2
using Printf, CairoMakie

# 1) Make sure OUTPUTS/ exists
mkpath(joinpath(@__DIR__, "OUTPUTS"))

# 2) Build the shallow-water model
@info "Set up model"
Nx, Ny = 64, 32          # ← test at low res, then bump to 1024×256
Lx, Ly = 2π, 20.0

grid = RectilinearGrid(
  size     = (Nx, Ny),
  x        = (0, Lx),
  y        = (-Ly/2, Ly/2),
  topology = (Periodic, Bounded, Flat),
)

g = 9.81
model = ShallowWaterModel(
  grid                     = grid,
  gravitational_acceleration = g,
  timestepper              = :RungeKutta3,
  # For a conservative formulation you could add:
  # formulation = ConservativeFormulation(),
)

# 3) Prescribe your dipole initial conditions directly into uh, vh, h
@info "Set initial conditions"
A₀, α₀, x₀, y₀, H = 1.0, 1.0, π, 0.5, 15.0

ψ(x,y) = A₀ * exp(-α₀*((x - x₀)^2 + (y - y₀)^2)) +
         A₀ * exp(-α₀*((x - x₀)^2 + (y + y₀)^2))

# From streamfunction ψ → velocities u, v
uₛ(x,y) =  ∂y(ψ)(x,y)
vₛ(x,y) = -∂x(ψ)(x,y)

# Set height = H everywhere, momentum = uₛ*H, vₛ*H
set!(model,
     h  = (x,y) -> H,
     uh = (x,y) -> uₛ(x,y)*H,
     vh = (x,y) -> vₛ(x,y)*H)

# 4) Define diagnostics _without_ dividing mismatched arrays
@info "Setting up fields"
u = model.velocities.u
v = model.velocities.v

ω = Field(∂x(v) - ∂y(u))        # vorticity on the z-axis
s = Field(sqrt(u^2 + v^2))      # speed magnitude

# 5) Simulation + I/O
@info "Set up simulation"
simulation = Simulation(model, Δt = 1e-4, stop_time = 12)

# CFL wizard + progress every 500 iters
wizard = TimeStepWizard(cfl=0.7, max_change=1.1, max_Δt=1e-4)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(500))

function progress_message(sim)
  max_u = maximum(abs, sim.model.velocities.u)
  max_v = maximum(abs, sim.model.velocities.v)
  wt    = prettytime(sim.run_wall_time)
  @info @sprintf("Iter %06d, t=%.3f, Δt=%.2e, max|u|=%.2e, max|v|=%.2e, wall=%s",
                 iteration(sim), time(sim), sim.Δt, max_u, max_v, wt)
end
add_callback!(simulation, progress_message, IterationInterval(500))

fields_file  = joinpath(@__DIR__, "OUTPUTS", "dipole_fields.jld2")
height_file  = joinpath(@__DIR__, "OUTPUTS", "dipole_height.jld2")

simulation.output_writers[:fields] = JLD2OutputWriter(model, (ω, s);
  schedule           = TimeInterval(0.5),
  filename           = fields_file,
  overwrite_existing = true,
)

simulation.output_writers[:height] = JLD2OutputWriter(model, (h,);
  schedule           = TimeInterval(0.5),
  filename           = height_file,
  overwrite_existing = true,
)

@info "Run the simulation"
run!(simulation)

@info "Load data from JLD2 files"
ω_ts = FieldTimeSeries(fields_file, "ω")
s_ts = FieldTimeSeries(fields_file, "s")
h_ts = FieldTimeSeries(height_file, "h")
times = ω_ts.times
println("Saved times: ", times)

# 6) Plot + animate (inside a function to avoid REPL scope warnings)
function make_dipole_plots()
  x, y = xnodes(grid), ynodes(grid)

  # precompute color ranges
  ωlims = (minimum(interior(ω_ts)), maximum(interior(ω_ts)))
  slims = (minimum(interior(s_ts)), maximum(interior(s_ts)))
  hlims = (minimum(interior(h_ts)), maximum(interior(h_ts)))

  fig = Figure(resolution=(1200,1600), fontsize=32)

  fontsize = 28
  axis_kwargs = (
    xlabel="x", ylabel="y",
    xlabelsize=fontsize, ylabelsize=fontsize,
    xticklabelsize=fontsize, yticklabelsize=fontsize,
    xticks=(0:π/3:2π, ["0","π/3","2π/3","π","4π/3","5π/3","2π"]),
    yticks=-10:2:10,
    limits=((0,2π),(-10,10)),
    titlefontsize=fontsize
  )

  n = Observable(1)
  ω_field = @lift ω_ts[$n]
  s_field = @lift s_ts[$n]
  h_field = @lift h_ts[$n]

  function add_row(row, title, field, crange, cmap, label)
    ax = Axis(fig[row,1]; title=title, axis_kwargs...)
    hm = heatmap!(ax, x, y, field; colormap=cmap, colorrange=crange)
    Colorbar(fig[row,2], hm; label=label, labelsize=20, ticklabelsize=20)
  end

  add_row(2, L"Vorticity $ω$",      ω_field, ωlims, :balance, L"$s^{-1}$")
  add_row(3, L"Speed $|\mathbf v|$", s_field, slims, :speed, L"m/s")
  add_row(4, L"Height $h$",         h_field, hlims, :balance, L"m")

  fig[1,:] = Label(fig, @lift @sprintf("t = %.1f", times[$n]),
                   fontsize=24, tellwidth=false)

  save(joinpath(@__DIR__, "OUTPUTS", "dipole_vorticity.png"), fig)
  record(fig,
         joinpath(@__DIR__, "OUTPUTS", "dipole_vorticity.mp4"),
         1:length(times); framerate=8) do i
    n[] = i
  end
end

make_dipole_plots()