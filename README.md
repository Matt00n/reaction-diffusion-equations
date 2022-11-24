# Reaktion-diffusion problems

Implementation of numerical methods to approximately solve specific reaction-diffusion equations
of the form $\varepsilon^2 u''(x) + c(x)*u(x) = f(x), \quad u(0) = u(1) = 0$, where 
$c, f \colon I \coloneqq [0, 1] \rightarrow \mathbb{R}$ continuous with 
$\min_{x \in I} c(x) = \gamma^2, \: \gamma > 0$ and $\varepsilon \in \mathbb{R}, \: 0 < \varepsilon \ll 1$.

The solver uses a discretization based a mesh and finite difference methods to numerically 
approximate the solution to reaction-diffusion problems. The arguments of the `solve` method
gives different options for how to do this. 

- `shishkin_mesh=True` enables the use of a piecewise-linear Shishkin mesh instead of 
a uniform mesh. The Shishkin mesh is defined as follows: Let 
$\tau \coloneqq \min \{\frac{1}{4}, \frac{\sigma \varepsilon}{\gamma}\ln N\}, \: \sigma \geq 2$,
where $N+1$ is the number of mesh points. The intervals $[0, \tau]$ and $[1- \tau, 1]$ are
divided in $N/4$ subintervals respectively while $[\tau, 1- \tau]$ is divided in $N/2$.
$\sigma$ is a tunable mesh parameter which can be specified with the `sigma` argument.
- By default, the solver uses the following finite differnce scheme:
$\bigl(Lu^N\bigr)(x_i) \coloneqq -\varepsilon^2\Bigl(\delta^2_hu^N\Bigr)(x_i) + c(x_i)u^N(x_i) = f(x_i) \qquad \text{für} \ i = 1, \hdots, N-1$,
where $\Bigl(\delta^2_hg\Bigr)(x_i) \coloneqq \frac{2}{h_i+h_{i+1}} \biggl(\frac{g(x_{i+1})-g(x_{i})}{h_{i+1}} - \frac{g(x_{i})-g(x_{i-1})}{h_{i}}\biggr)$. With `advanced_solve=True`, scheme is replaced with a modified scheme:
$Bigl(L^mu^N\Bigr)(x_i) = \bigl(Qf\bigr)(x_i), \ i = 1, \hdots, N-1$, where 
$Bigl(L^mu^N\Bigr)(x_i) \coloneqq -\varepsilon^2\bigl(\delta^2_hu^N\bigr)(x_i) + \mu^+_i(cu)(x_{i+1}) + \mu^0_i(cu)(x_{i}) + \mu^-_i(cu)(x_{i-1})$
and $\bigl(Qf\bigr)(x_i) \coloneqq  \nu^+_if(x_{i+1}) + \nu^0_if(x_{i}) + \nu^-_if(x_{i-1})$. The coefficients
$\mu^*_*$ and $\nu^*_*$ are determined such that $\bigl(L^mp\bigr)(x_i) = \bigl(Q(Lp)\bigr)(x_i)$ for all $p \in \Pi_5$.
- `double_mesh=True` enables the calculation of a reference solution based on the 
double mesh principle which is returned in addition to the original approximation.

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


