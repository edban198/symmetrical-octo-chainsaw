using Oceananigans
using JLD2
using Printf, CairoMakie

@info "Set up model"

Nx, Ny = 64, 32
Lx = 2π
Ly = 20

grid = RectilinearGrid(size=(Nx, Ny),
                       x=(0, Lx),
                       y=(-Ly/2, Ly/2),
                       topology=(Periodic, Bounded, Flat))

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
ω = Field(∂x(model.velocities.v) - ∂y(model.velocities.u))
v_norm = Field(sqrt.(model.velocities.u^2 + model.velocities.v^2))
compute!(ω)
compute!(v_norm)

@info "Set up simulation"
simulation = Simulation(model, Δt=1e-2, stop_time=10)

@info "Set up output writers"
fields_filename = joinpath(@__DIR__, "dipole_shallow_water_fields.jld2")
simulation.output_writers[:fields] = JLD2OutputWriter(model, (; ω, v_norm, h),
                                                      filename=fields_filename,
                                                      schedule=TimeInterval(1),
                                                      overwrite_existing=true)

@info "Run the simulation"
run!(simulation)

# Visualization:
@info "Plot graphs and make animations"

x, y = xnodes(ω), ynodes(ω)

fig = Figure(size=(1200, 1600), fontsize=20)

axis_kwargs = (xlabel="x", ylabel="y")

ax_ω = Axis(fig[2, 1]; title=L"Vorticity, $ω$", axis_kwargs...)
ax_v_norm = Axis(fig[3, 1]; title=L"Velocity magnitude, $|\mathbf{v}|$", axis_kwargs...)
ax_h = Axis(fig[4, 1]; title=L"Height, $h$", axis_kwargs...)

n = Observable(1)

@info "Load data from JLD2 file"
data = jldopen(fields_filename, "r") do file
    times = file["times"]
    ω_data = file["ω"]
    v_norm_data = file["v_norm"]
    h_data = file["h"]
    (times, ω_data, v_norm_data, h_data)
end

times, ω_data, v_norm_data, h_data = data

hm_ω = heatmap!(ax_ω, x, y, ω_frame, colorrange=(-1, 1), colormap=:balance)
Colorbar(fig[2, 2], hm_ω)

hm_v_norm = heatmap!(ax_v_norm, x, y, v_norm_frame, colorrange=(0, 1), colormap=:balance)
Colorbar(fig[3, 2], hm_v_norm)

hm_h = heatmap!(ax_h, x, y, h_frame, colormap=:balance)
Colorbar(fig[4, 2], hm_h)

#title = L"Total vorticity, ω from a dipole from the streamfunction $\Psi = A_{0}e^{-\alpha_{0}((x - x_{0})^2 + (y - y_{0})^2)} + A_{0}e^{-\alpha_{0}((x - x_{0})^2 + (y + y_{0})^2)}$ with $(x_{0},y_{0})=(\pi,0)$"
title = @lift @sprintf("t = %.1f", times[$n])
fig[1, :] = Label(fig, title, fontsize=24, tellwidth=false)

@info "Save plots and record animation"
save("dipole_shallow_water_total_vorticity.png", fig)

frames = 1:length(times)
record(fig, "dipole_shallow_water_total_vorticity_animation.mp4", frames, framerate=12) do i
    n[] = i
end

# Define a grid
x = LinRange(0, 2π, 16)
y = LinRange(-10, 10, 16)
X, Y = [xi for xi in x, _ in y], [yi for _ in x, yi in y]

# Parameters
A₀ = 1
α₀ = 1
x₀ = π
y₀ = 0.5

# Define U(x, y) and V(x, y)

# Evaluate U and V
U_vals = u₀.(X, Y)
V_vals = v₀.(X, Y)

# Create figure
fig = Figure(resolution=(1200, 800))
ax = Axis(fig[1, 1]; limits = ((0, 2π), (-10, 10)),
          xlabel = "x", ylabel = "y", title = "Velocity Field with Arrows"
)

# Add arrows
arrows!(ax, vec(X), vec(Y), vec(U_vals), vec(V_vals))

# Save figure
save("background_velocity_field.png", fig)