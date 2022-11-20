# Reaktion-diffusion problems

Implementation of numerical methods to solve specific reaction-diffusion equations
of the form $\varepsilon^2 u''(x) + c(x)*u(x) = f(x)$.

The solver uses a discretization based on uniform and Shishkin meshes in combination
with finite difference methods to numerically approximate the solution to reaction-
diffusion problems.

## Usage:

```
# create reaction-diffusion problem instance
f = lambda x: np.exp(x-1) 
c = 4
rde = ReactionDiffusionEquation(f=f, c=4)


# compute approximate solutions
x, u = rde.solve(eps=0.1, n=17, shishkin_mesh=True, sigma=4, 
                advanced_solve=True, verbose=False)

# plot approximations
x, u = plot_solve(eps=0.1, n=17, shishkin_mesh=True, sigma=4, 
                advanced_solve=True, interpolation='linear')
```