using Oceananigans
using JLD2
using Printf, CairoMakie

@info "Set up model"

Nx, Ny = 1024, 256
Lx = 2π
Ly = 20

grid = RectilinearGrid(size=(Nx, Ny),
                       x=(0, Lx),
                       y=(-Ly/2, Ly/2),
                       topology=(Periodic, Bounded, Flat)
)

h₀ = 1
xₕ = 7π/4
yₕ = 0
σ = 2

bottom(x, y) = h₀ * exp(-σ * ((x - xₕ)^2 + (y - yₕ)^2))
gravitational_acceleration = 9.81

# Model:
model = ShallowWaterModel(; grid, gravitational_acceleration, timestepper=:RungeKutta3)

@info "Set initial conditions"
uh, vh, h = model.solution

u = uh / h
v = vh / h

A₀, α₀, x₀, y₀, H = 1, 1, π, 0.5, 15

uᵢ(x, y) = A₀ * 2 * (y - y₀) * α₀ * exp(-α₀ * ((x - x₀)^2 + (y - y₀)^2)) -
            A₀ * 2 * (y + y₀) * α₀ * exp(-α₀ * ((x - x₀)^2 + (y + y₀)^2))
vᵢ(x, y) = -(A₀ * 2 * (x - x₀) * α₀ * exp(-α₀ * ((x - x₀)^2 + (y - y₀)^2)) -
              A₀ * 2 * (x - x₀) * α₀ * exp(-α₀ * ((x - x₀)^2 + (y + y₀)^2)))

h̄(x, y) = H

uhᵢ(x, y) = uᵢ(x, y) * h̄(x, y)
vhᵢ(x, y) = vᵢ(x, y) * h̄(x, y)

set!(model, uh=uhᵢ, vh=vhᵢ, h=h̄)

@info "Setting up fields"
ω = Field(∂x(v) - ∂y(u))
s = Field(sqrt(u^2 + v^2))

@info "Set up simulation"
simulation = Simulation(model, Δt=1e-4, stop_time=12)

@info "Set up progress message and timestep wizard"
wizard = TimeStepWizard(cfl=0.7, max_change=1.1, max_Δt=1e-4)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(100))

function progress_message(sim)
    max_abs_u = maximum(abs, sim.model.velocities.u)
    max_abs_v = maximum(abs, sim.model.velocities.v)
    walltime = prettytime(sim.run_wall_time)

    return @info @sprintf("Iteration: %04d, time: %1.3f, Δt: %.2e, max(|u|) = %.1e, max(|v|) = %.1e, wall time: %s\n",
                          iteration(sim), time(sim), sim.Δt, max_abs_u, max_abs_v, walltime)
end

add_callback!(simulation, progress_message, IterationInterval(100))

@info "Set up output writers"
fields_filename = joinpath(@__DIR__, "./OUTPUTS/dipole_shallow_water_fields")
heights_filename = joinpath(@__DIR__, "./OUTPUTS/dipole_shallow_water_heights")
simulation.output_writers[:fields] = JLD2OutputWriter(model, (; ω, s),
                                                      schedule = TimeInterval(0.5),
                                                      filename = fields_filename * ".jld2",
                                                      overwrite_existing = true
)

simulation.output_writers[:height] = JLD2OutputWriter(model, (; h),
                                                      schedule = TimeInterval(0.5),
                                                      filename = heights_filename * ".jld2",
                                                      overwrite_existing = true)

@info "Run the simulation"
run!(simulation)

@info "Load data from JLD2 file"
ω_timeseries = FieldTimeSeries(fields_filename * ".jld2", "ω")
s_timeseries = FieldTimeSeries(fields_filename * ".jld2", "s")
h_timeseries = FieldTimeSeries(heights_filename * ".jld2", "h")

times = ω_timeseries.times
println("Saved times: ", ω_timeseries.times)

# Visualization:
@info "Plot graphs and make animations"

x, y = xnodes(ω), ynodes(ω)

fig = Figure(size=(1200, 1600), fontsize=20)

axis_kwargs = (xlabel="x", ylabel="y")

ax_ω = Axis(fig[2, 1]; title=L"Vorticity, $ω$", axis_kwargs...)
ax_s = Axis(fig[3, 1]; title=L"Velocity magnitude, $|\mathbf{v}|$", axis_kwargs...)
ax_h = Axis(fig[4, 1]; title=L"Height, $h$", axis_kwargs...)

n = Observable(1)

ω = @lift ω_timeseries[$n]
s = @lift s_timeseries[$n]
h = @lift h_timeseries[$n]

ωlims = (minimum(interior(ω_timeseries)), maximum(interior(ω_timeseries)))
slims = (minimum(interior(s_timeseries)), maximum(interior(s_timeseries)))
hlims = (minimum(interior(h_timeseries)), maximum(interior(h_timeseries)))

hm_ω = heatmap!(ax_ω, x, y, ω, colormap=:balance, colorrange=ωlims)
Colorbar(fig[2, 2], hm_ω)

hm_s = heatmap!(ax_s, x, y, s, colormap=:speed, colorrange=slims)
Colorbar(fig[3, 2], hm_s)

hm_h = heatmap!(ax_h, x, y, h, colormap=:balance, colorrange=hlims)
Colorbar(fig[4, 2], hm_h)

#title = L"Total vorticity, ω from a dipole from the streamfunction $\Psi = A_{0}e^{-\alpha_{0}((x - x_{0})^2 + (y - y_{0})^2)} + A_{0}e^{-\alpha_{0}((x - x_{0})^2 + (y + y_{0})^2)}$ with $(x_{0},y_{0})=(\pi,0)$"
title = @lift @sprintf("t = %.1f", times[$n])
fig[1, :] = Label(fig, title, fontsize=24, tellwidth=false)

@info "Save plots and record animation"
save("./OUTPUTS/dipole_shallow_water_total_vorticity.png", fig)

frames = 1:length(times)
record(fig, "./OUTPUTS/dipole_shallow_water_total_vorticity_animation.mp4", frames, framerate=8) do i
    n[] = i
end

# Define a grid
x = LinRange(π\3, 5π/3, 20)
y = LinRange(-4, 4, 20)
X, Y = [xi for xi in x, _ in y], [yi for _ in x, yi in y]

# Evaluate U and V
U_vals = uᵢ.(X, Y)
V_vals = vᵢ.(X, Y)

U_vals_scaled = U_vals ./ (maximum(abs.(U_vals))*2)
V_vals_scaled = V_vals ./ (maximum(abs.(V_vals))*2)
# Create figure
fig = Figure(size=(1200, 800))
ax = Axis(fig[1, 1]; limits = ((π\3, 5π/3), (-4, 4)),
          xlabel = "x", ylabel = "y",
          xlabelsize = 24, ylabelsize = 24,
          xticklabelsize = 20, yticklabelsize = 20,
          xticks = (π\3:π/3:5π/3, ["π/3", "2π/3", "π", "4π/3", "5π/3"]), yticks = (-4:2:4),
          title = L"Velocity Field of $\Psi$", titlesize = 24
)

# Add arrows
arrows!(ax, vec(X), vec(Y), vec(U_vals_scaled), vec(V_vals_scaled))

# Save figure
save("./OUTPUTS/velocity_field.png", fig)

#Timesnaps plot

len = length(times)
values = collect(range(1, stop=len, length=4))

selected_indices = round.(Int, values)[1:4]
selected_times = times[selected_indices]

fig = Figure(size=(3400, 1600))
Label(fig[1, 1:4], "Snapshots of the evolution of a dipole in shallow water", fontsize=64)

axis_kwargs_2 = (xlabel="x", ylabel="y",
                 xlabelsize=42, ylabelsize=42,
                 xticklabelsize=32, yticklabelsize=32,
                 limits=((0, 2π), (-10, 10)))

#=
# Add row titles (properly aligned in their own space)
Label(fig.layout[2, 0], L"Vorticity $ω$", rotation=π/2, fontsize=42, halign=:center)
Label(fig.layout[3, 0], L"Velocity Magnitude $|v|$", rotation=π/2, fontsize=42, halign=:center)
Label(fig.layout[4, 0], L"Height $h$", rotation=π/2, fontsize=42, halign=:center)
=#
# Plot timesnaps
for (i, idx) in enumerate(selected_indices)
    ω_snapshot = ω_timeseries[idx]
    s_snapshot = s_timeseries[idx]
    h_snapshot = h_timeseries[idx]

    ax_ω = Axis(fig[2, i]; title=@sprintf("t = %.1f", selected_times[i]), titlesize=42, axis_kwargs_2...)
    heatmap!(ax_ω, x, y, ω_snapshot, colormap=:balance, colorrange=ωlims)

    ax_s = Axis(fig[3, i]; axis_kwargs_2...)
    heatmap!(ax_s, x, y, s_snapshot, colormap=:speed, colorrange=slims)

    ax_h = Axis(fig[4, i]; axis_kwargs_2...)
    heatmap!(ax_h, x, y, h_snapshot, colormap=:balance, colorrange=hlims)
end

# Add colorbars with larger ticks and labels
Colorbar(fig[2, 5]; colormap=:balance, colorrange=ωlims, ticklabelsize=32, labelsize=32, label = L"Vorticity $ω$")
Colorbar(fig[3, 5]; colormap=:speed, colorrange=slims, ticklabelsize=32, labelsize=32, label = L"Velocity Magnitude $|v|$")
Colorbar(fig[4, 5]; colormap=:balance, colorrange=hlims, ticklabelsize=32, labelsize=32, label = L"Height $h$")

# Save the figure
save("./OUTPUTS/dipole_shallow_water_snapshots.png", fig)