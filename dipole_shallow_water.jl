using Oceananigans

@info"set model"

Nx, Ny = 64, 32
#Nx, Ny = 256, 128

Lx = 2π
Ly = 20

grid = RectilinearGrid(size = (Nx, Ny),
                       x = (0, Lx),
                       y = (-Ly/2, Ly/2),
                       topology = (Periodic, Bounded, Flat)
)

h₀ = 1
xₕ = 7π/4
yₕ = 0
σ = 2

bottom(x, y) = h₀ * exp(-σ * ((x-xₕ)^2 + (y-yₕ)^2))

#gravitational_acceleration = 1



# Model:
model = ShallowWaterModel(; grid, gravitational_acceleration,
                          timestepper = :RungeKutta3
)


@info"setting intial conditions"
# Set initial conditions:
uh, vh, h = model.solution

u = uh / h
v = vh / h

ω = Field(∂x(v) - ∂y(u))
compute!(ω)

A₀ = 1
α₀ = 1
x₀ = π
y₀ = 0.5

using LaTeXStrings

stream_func = L"$\Psi = A_{0}e^{-\alpha_{0}((x - x_{0})^2 + (y - y_{0})^2)} + A_{0}e^{-\alpha_{0}((x - x_{0})^2 + (y + y_{0})^2)}$"

uᵢ(x,y) = A₀ * 2 * (y - y₀) * α₀ * exp(-α₀ * ((x - x₀)^2 + (y - y₀)^2)) - A₀ * 2 * (y + y₀) * α₀ * exp(-α₀ * ((x - x₀)^2 + (y + y₀)^2))
vᵢ(x,y) = - (A₀ * 2 * (x - x₀) * α₀ * exp(-α₀ * ((x - x₀)^2 + (y - y₀)^2)) - A₀ * 2 * (x - x₀) * α₀ * exp(-α₀ * ((x - x₀)^2 + (y + y₀)^2)))

H = 15

h̄(x,y) = H #set inital constant height

uhᵢ(x,y) = uᵢ(x,y) * h̄(x,y)
vhᵢ(x,y) = vᵢ(x,y) * h̄(x,y)

set!(model, uh = uhᵢ, vh = vhᵢ, h = h̄)

simulation = Simulation(model, Δt = 1e-2, stop_time = 10)

@info"setting up output writers"
# Set up output writers:
fields_filename = joinpath(@__DIR__, "dipole_shallow_water_fields.nc")
simple_outputs = ()
simulation.output_writers[:fields] = NetCDFOutputWriter(model, (; ω),
                                                        filename = fields_filename,
                                                        schedule = TimeInterval(1),
                                                        overwrite_existing = true
)

#|u|?

@info"run the simulation"
run!(simulation)

# Visualise:

@info"plot graph and make animation"

using NCDatasets, Printf, CairoMakie

x, y = xnodes(ω), ynodes(ω)

fig = Figure(size = (1200,800), fontsize = 20)

axis_kwargs = (xlabel = "x", ylabel = "y")
title = L"Total vorticity, ω from a dipole from the streamfunction $\Psi = A_{0}e^{-\alpha_{0}((x - x_{0})^2 + (y - y_{0})^2)} + A_{0}e^{-\alpha_{0}((x - x_{0})^2 + (y + y_{0})^2)}$ with $(x_{0},y_{0})=(\pi,0)$"
ax_ω  = Axis(fig[1, 1]; title, axis_kwargs...)

n = Observable(1)

ds = NCDataset(simulation.output_writers[:fields].filepath, "r")

times = ds["time"][:]

ω = @lift ds["ω"][:, :, 1, $n]
hm_ω = heatmap!(ax_ω, x, y, ω, colorrange=(-1, 1), colormap=:balance)
Colorbar(fig[1, 2], hm_ω)

title = @lift @sprintf("t = %.1f", times[$n])
fig[0, :] = Label(fig, title, fontsize=24, tellwidth=false)

#text!(fig[1,1], 0.5, -8, text = "Coriolis frequency = 1e-4", color = :black)

CairoMakie.activate!(type = "png")

save("dipole_shallow_water_total_vorticity.png", fig)

# Define the frames for recording and record the animation:
frames = 1:length(times)

record(fig, "dipole_shallow_water_total_vorticity_animation.mp4", frames, framerate=12) do i
    n[] = i
end

close(ds)

using CairoMakie

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
          xlabel = "x", ylabel = "y", title = "Velocity Field with Arrows")

# Add arrows
arrows!(ax, vec(X), vec(Y), vec(U_vals), vec(V_vals))

# Save figure
save("velocity_field.png", fig)