import numpy as np
from typing import List, Optional, Tuple, Callable, Union
from scipy.optimize import basinhopping
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
import math


############# TODO #############
# TODO setup notebook to generate all tables & figures
# TODO Docstrings & README
# TODO (OPTIONAL) benchmark against to SymPy
# TODO cleanup on aisle 5
# TODO check format of multi-plotters --> modify tight layout rect
# NOTE: multiple solvers do not support reference solution !


def thomas_algorithm(
    e: np.ndarray,
    d: np.ndarray,
    f: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Thomas-Algorithmus
    """
    n = len(b) # get n
    u = np.zeros(n, np.float64) # initialize solution array

    # elimination
    for i in range(1, n):
        l = e[i] / d[i-1] 
        d[i] = d[i] - l * f[i-1]
        b[i] = b[i] - l * b[i-1]

    # backward substitution
    u[n-1] = b[n-1] / d[n-1] 
    for i in range(1, n):
        ind = n - i - 1
        u[ind] = (b[ind] - f[ind] * u[ind+1]) / d[ind] 

    return u


class ReactionDiffusionEquation():
    """
    TODO
    """
    def __init__(
        self,
        f: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
        c: Union[int, float, Callable[[Union[float, np.ndarray]], Union[int, float, np.ndarray]]],
        gamma_iterations: int = 100,
        gamma_start: float = 0.5,
        gamma_init_step: float = 0.5,
    ):
        """
        TODO
        """

        self.f = f
        self.c = c

        self.gamma_iterations = gamma_iterations
        self.gamma_start = gamma_start
        self.gamma_init_step = gamma_init_step
        self.gamma = self._get_gamma()

    def solve(
        self,
        eps: float = 1, 
        n: int = 3, 
        shishkin_mesh: bool = False,
        sigma: float = 2,
        advanced_solve: bool = False,
        double_mesh: bool = False,
        verbose: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], 
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
        """
        TODO
        """
        # NOTE: here n is the total number of mesh points, i.e. n = N + 1

        # assert n >= 3
        if n < 3:
            raise ValueError("n must be greater than 2")
        if shishkin_mesh and sigma < 2:
            raise ValueError("sigma must be greater equal than 2 when using a Shishkin mesh")
        if shishkin_mesh and sigma < 4 and advanced_solve:
            raise ValueError("sigma must be greater equal than 4 when using a Shishkin mesh with advanced_solve")

        if shishkin_mesh:
            # increase N if needed such that N is divisible by 4
            if (n-1) % 4 != 0:
                n = 4 * ((n-1) // 4 + 1) + 1  
                if verbose:
                    print(f'a Shishkin mesh requires n-1 to be divisible by 4, n adjusted to {n}')

            # compute shishkin mesh
            tau = np.minimum(1/4, np.log(n-1) * sigma * eps / self.gamma)
            if verbose:
                print(f'tau = {tau}')
            h1 = np.ones(int((n-1) / 4), np.float64) * tau / ((n-1) / 4)
            h2 = np.ones(int((n-1) / 2), np.float64) * (1 - 2 * tau) / ((n-1) / 2)
            h = np.concatenate([h1, h2, h1]) # fuse piecewise uniform meshes
        else:
            h = np.ones(n-1, np.float64) * 1/(n-1) # uniform mesh

        # get x values of mesh
        x = np.zeros(n, np.float64)
        x[1:] = np.cumsum(h) 
        # computation of x from mesh distances may cause rounding errors
        # hence we set the last value to 1 manually if needed
        if x[-1] != 1:
            if verbose:
                print(f'x_N adjusted from {x[-1]} to 1 due to rounding errors')
            x[-1] = 1

        # get the tridiagonal system of linear equations 
        e, d, f, b = self._init_sle(x=x, h=h, eps=eps, n=n, advanced_solve=advanced_solve)

        if verbose:
            self._print_sle(n=n, e=e, d=d, f=f, b=b) # print system of equations

        u = thomas_algorithm(e=e, d=d, f=f, b=b) # solve

        if double_mesh:
            # get the reference solution based on the double mesh principle
            x_ref, u_ref = self._get_reference_solution(x, eps=eps, n=n, advanced_solve=advanced_solve)
            return (x, u), (x_ref, u_ref)
        else:
            return x, u


    def plot_solve(
        self,
        eps: float = 1, 
        n: int = 3, 
        shishkin_mesh: bool = False,
        sigma: float = 2,
        advanced_solve: bool = False,
        verbose: bool = False,
        interpolation: Optional[str] = None,
        scatter: bool = True,
        support: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        # get solution
        x, u = self.solve(eps=eps, n=n, shishkin_mesh=shishkin_mesh, sigma=sigma, 
                            advanced_solve=advanced_solve, verbose=verbose)

        # plotting
        plt.figure(figsize=(16,10))
        if scatter:
            sns.scatterplot(x=x, y=u, color='black')

        # interpolate approximations at mesh points
        if interpolation == 'linear':
            sns.lineplot(x=x, y=u, linewidth=3, color='orange')
        elif interpolation == 'quadratic':
            f = interp1d(x, u, kind=interpolation)
            xnew = np.linspace(0., 1., 100)
            ynew = f(xnew)   
            plt.plot(xnew, ynew, '-', linewidth=3, color='orange')
        elif interpolation == 'cubic':
            if n < 4:
                raise ValueError('n must be greater than 3 for cubic interpolation')
            f = interp1d(x, u, kind=interpolation)
            xnew = np.linspace(0., 1., 100)
            ynew = f(xnew) 
            plt.plot(xnew, ynew, '-', linewidth=3, color='orange')

        if support:
            for j in x:
                plt.axvline(x=[j], color='lightgray')
        else:
            plt.grid()

        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title(f'Approximation of u(x) with eps = {eps} and a {"Shishkin" if shishkin_mesh else "uniform"} \
mesh with {n} mesh points', fontsize=20)
        plt.show()

        return x, u


    def solve_multiple_eps(
        self,
        eps: Union[List, np.ndarray], 
        n: int = 3, 
        shishkin_mesh: bool = False,
        sigma: float = 2,
        advanced_solve: bool = False,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # increase N to be divisible by 4 if using a Shishkin mesh
        if shishkin_mesh:
            if (n-1) % 4 != 0:
                n = 4 * ((n-1) // 4 + 1) + 1  
        solutions = []

        # get approximations for each epsilon
        for e in eps:
            x, u = self.solve(eps=e, n=n, shishkin_mesh=shishkin_mesh, sigma=sigma, 
                            advanced_solve=advanced_solve, verbose=verbose)
            solutions.append((x, u))
        return solutions
            

    def plot_solve_multiple_eps(
        self,
        eps: Union[List, np.ndarray], 
        n: int = 3, 
        shishkin_mesh: bool = False,
        sigma: float = 2,
        advanced_solve: bool = False,
        verbose: bool = False,
        interpolation: Optional[str] = None,
        subplots: bool = True,
        scatter: bool = True,
        support: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # get solutions
        solutions = self.solve_multiple_eps(eps=eps, n=n, shishkin_mesh=shishkin_mesh, sigma=sigma, 
                                            advanced_solve=advanced_solve, verbose=verbose)
        
        # plotting
        if subplots:
            height = 5 * math.ceil(len(eps)/2)
            fig, axes = plt.subplots(math.ceil(len(eps)/2), 2, figsize=(16,height), tight_layout=True)
            for i, ((x, u), ax) in enumerate(zip(solutions, axes.flatten())):
                
                if scatter:
                    sns.scatterplot(x=x, y=u, color='black', ax=ax)

                # interpolate approximations at mesh points
                if interpolation == 'linear':
                    sns.lineplot(x=x, y=u, linewidth=3, color='orange', ax=ax)
                elif interpolation == 'quadratic':
                    f = interp1d(x, u, kind=interpolation)
                    xnew = np.linspace(0., 1., 100)
                    ynew = f(xnew) 
                    ax.plot(xnew, ynew, '-', linewidth=3, color='orange')

                elif interpolation == 'cubic':
                    if n < 4:
                        raise ValueError('n must be greater than 3 for cubic interpolation')
                    f = interp1d(x, u, kind=interpolation)
                    xnew = np.linspace(0., 1., 100)
                    ynew = f(xnew) 
                    ax.plot(xnew, ynew, '-', linewidth=3, color='orange')

                if support:
                    for j in x:
                        ax.axvline(x=[j], color='lightgray')
                else:
                    ax.grid()

                ax.set_title(f'epsilon = {eps[i]}')
                ax.set_xlabel('x')
                ax.set_ylabel('u(x)')

            fig.suptitle(f'Approximation of u(x) with a {"Shishkin" if shishkin_mesh else "uniform"} \
mesh \n with {n} mesh points for various values of epsilon', fontsize=20)
            plt.show()

        else:
            plt.figure(figsize=(16,10))
            for i, (x, u) in enumerate(solutions):
                
                if scatter:
                    sns.scatterplot(x=x, y=u, color='black')

                # interpolate approximations at mesh points
                if interpolation == 'linear':
                    sns.lineplot(x=x, y=u, linewidth=2, label=f'eps = {eps[i]}')
                elif interpolation == 'quadratic':
                    f = interp1d(x, u, kind=interpolation)
                    xnew = np.linspace(0., 1., 100)
                    ynew = f(xnew)  
                    plt.plot(xnew, ynew, '-', linewidth=2, label=f'eps = {eps[i]}')

                elif interpolation == 'cubic':
                    f = interp1d(x, u, kind=interpolation)
                    xnew = np.linspace(0., 1., 100)
                    ynew = f(xnew)
                    plt.plot(xnew, ynew, '-', linewidth=2, label=f'eps = {eps[i]}')

            if support and not shishkin_mesh:
                for j in x:
                    plt.axvline(x=[j], color='lightgray')
            else:
                plt.grid()

            plt.legend()
            plt.xlabel('x')
            plt.ylabel('u(x)')
            plt.title(f'Approximation of u(x) with a {"Shishkin" if shishkin_mesh else "uniform"} \
mesh with {n} mesh points \n for various values of epsilon', fontsize=20)
            plt.show()

        return x, solutions


    def solve_multiple_n(
        self,
        n: Union[List, np.ndarray],
        eps: float = 1,
        shishkin_mesh: bool = False,
        sigma: float = 2,
        advanced_solve: bool = False,
        verbose: bool = False,
    ) -> List:
        # get approximations for each n
        solutions = []
        for i in n:
            x, u = self.solve(eps=eps, n=i, shishkin_mesh=shishkin_mesh, sigma=sigma, 
                            advanced_solve=advanced_solve, verbose=verbose)
            solutions.append((x, u))
        return solutions


    def plot_solve_multiple_n(
        self,
        n: Union[List, np.ndarray],
        eps: float = 1,
        shishkin_mesh: bool = False,
        sigma: float = 2,
        advanced_solve: bool = False,
        verbose: bool = False,
        interpolation: Optional[str] = None,
        subplots: bool = True,
        scatter: bool = True,
    ) -> List:
        # get solutions for each n
        solutions = self.solve_multiple_n(eps=eps, n=n, shishkin_mesh=shishkin_mesh, sigma=sigma, 
                                            advanced_solve=advanced_solve, verbose=verbose)
        # plotting
        if subplots:
            height = 5 * math.ceil(len(n)/2)
            fig, axes = plt.subplots(math.ceil(len(n)/2), 2, figsize=(16,height), tight_layout=True)
            for i, ((x, u), ax) in enumerate(zip(solutions, axes.flatten())):
                
                if scatter:
                    sns.scatterplot(x=x, y=u, color='black', ax=ax)

                # interpolate approximations at mesh points
                if interpolation == 'linear':
                    sns.lineplot(x=x, y=u, linewidth=3, color='orange', ax=ax)
                elif interpolation == 'quadratic':
                    f = interp1d(x, u, kind=interpolation)
                    xnew = np.linspace(0., 1., 100)
                    ynew = f(xnew)  
                    ax.plot(xnew, ynew, '-', linewidth=3, color='orange')

                elif interpolation == 'cubic':
                    if np.min(n) < 4:
                        raise ValueError('n must be greater than 3 for cubic interpolation')
                    f = interp1d(x, u, kind=interpolation)
                    xnew = np.linspace(0., 1., 100)
                    ynew = f(xnew) 
                    ax.plot(xnew, ynew, '-', linewidth=3, color='orange')

                ax.set_title(f'n = {n[i]}')
                ax.set_xlabel('x')
                ax.set_ylabel('u(x)')
                ax.grid()

            fig.suptitle(f'Approximation of u(x) with eps = {eps} and a {"Shishkin" if shishkin_mesh else "uniform"} \
mesh for various values of n', fontsize=20)
            plt.show()

        else:
            plt.figure(figsize=(16,10))
            for i, (x, u) in enumerate(solutions):
                
                if scatter:
                    sns.scatterplot(x=x, y=u, color='black')

                # interpolate approximations at mesh points
                if interpolation == 'linear':
                    sns.lineplot(x=x, y=u, linewidth=2, label=f'n = {n[i]}')
                elif interpolation == 'quadratic':
                    f = interp1d(x, u, kind=interpolation)
                    xnew = np.linspace(0., 1., 100)
                    ynew = f(xnew) 
                    plt.plot(xnew, ynew, '-', linewidth=2, label=f'n = {n[i]}')

                elif interpolation == 'cubic':
                    f = interp1d(x, u, kind=interpolation)
                    xnew = np.linspace(0., 1., 100)
                    ynew = f(xnew) 
                    plt.plot(xnew, ynew, '-', linewidth=2, label=f'n = {n[i]}')

            plt.legend()
            plt.xlabel('x')
            plt.ylabel('u(x)')
            plt.title(f'Approximation of u(x) with a {"Shishkin" if shishkin_mesh else "uniform"} \
mesh for various values of n', fontsize=20)
            plt.grid()
            plt.show()

        return solutions


    def plot_compare_solutions(
        self,
        n: int = 3,
        eps: float = 1,
        sigma: List = [2, 4, 4],
        interpolation: Optional[str] = None,
        subplots: bool = True,
        scatter: bool = True,
    ) -> None:
        # get approximations via each method
        x_base, u_base = self.solve(eps=eps, n=n, shishkin_mesh=False, sigma=sigma[0], 
                            advanced_solve=False, verbose=False)
        x_opt, u_opt = self.solve(eps=eps, n=n, shishkin_mesh=True, sigma=sigma[0], 
                            advanced_solve=False, verbose=False)
        x_adv, u_adv = self.solve(eps=eps, n=n, shishkin_mesh=False, sigma=sigma[1], 
                            advanced_solve=True, verbose=False)
        x_full, u_full = self.solve(eps=eps, n=n, shishkin_mesh=True, sigma=sigma[2], 
                            advanced_solve=True, verbose=False)
        solutions = [(x_base, u_base), (x_opt, u_opt), (x_adv, u_adv), (x_full, u_full)]
        titles = {0: 'base sol.', 1: 'opt. grid', 2: 'adv. sol.', 3: 'adv. sol. w/ opt. grid'}
        
        # plotting
        if subplots:
            fig, axes = plt.subplots(2, 2, figsize=(16,10), tight_layout=True)
            for i, ((x, u), ax) in enumerate(zip(solutions, axes.flatten())):
                
                if scatter:
                    sns.scatterplot(x=x, y=u, color='black', ax=ax)

                # interpolate approximations at mesh points
                if interpolation == 'linear':
                    sns.lineplot(x=x, y=u, linewidth=3, color='orange', ax=ax)
                elif interpolation == 'quadratic':
                    f = interp1d(x, u, kind=interpolation)
                    xnew = np.linspace(0., 1., 100)
                    ynew = f(xnew)  
                    ax.plot(xnew, ynew, '-', linewidth=3, color='orange')

                elif interpolation == 'cubic':
                    if np.min(n) < 4:
                        raise ValueError('n must be greater than 3 for cubic interpolation')
                    f = interp1d(x, u, kind=interpolation)
                    xnew = np.linspace(0., 1., 100)
                    ynew = f(xnew)  
                    ax.plot(xnew, ynew, '-', linewidth=3, color='orange')

                ax.set_title(titles[i])
                ax.set_xlabel('x')
                ax.set_ylabel('u(x)')
                ax.grid()

            fig.suptitle(f'Approximations of u(x) with eps = {eps} and {n} supporting points \
using different approaches', fontsize=20)
            plt.show()

        else:
            plt.figure(figsize=(16,10))
            for i, (x, u) in enumerate(solutions):
                
                if scatter:
                    sns.scatterplot(x=x, y=u, color='black')

                # interpolate approximations at mesh points
                if interpolation == 'linear':
                    sns.lineplot(x=x, y=u, linewidth=2, label=titles[i])
                elif interpolation == 'quadratic':
                    f = interp1d(x, u, kind=interpolation)
                    xnew = np.linspace(0., 1., 100)
                    ynew = f(xnew)   
                    plt.plot(xnew, ynew, '-', linewidth=2, label=titles[i])

                elif interpolation == 'cubic':
                    f = interp1d(x, u, kind=interpolation)
                    xnew = np.linspace(0., 1., 100)
                    ynew = f(xnew)  
                    plt.plot(xnew, ynew, '-', linewidth=2, label=titles[i])

            plt.legend()
            plt.xlabel('x')
            plt.ylabel('u(x)')
            plt.title(f'Approximations of u(x) with eps = {eps} and {n} supporting points \
using different approaches', fontsize=20)
            plt.grid()
            plt.show()


    def _get_gamma(self):
        """
        TODO
        """

        if not callable(self.c):
            # if constant coefficient
            return np.sqrt(self.c)
        else:
            # for variable coefficients
            # set up optimizer
            bounds = [(0., 1.)]
            minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds)
            take_step = BoundedOptimizerStep(stepsize=self.gamma_init_step)
            result = basinhopping(self.c, self.gamma_start, niter=self.gamma_iterations, minimizer_kwargs=minimizer_kwargs,
                        take_step=take_step)
            return np.sqrt(result.fun)
    
    
    def _get_reference_solution(
        self,
        x_old: np.ndarray,
        eps: float = 1,      
        n: int = 3, 
        advanced_solve: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        generates a reference solution using the double mesh principle.
        """
        
        # calculate new x by bisecting all intervals 
        x_mid_points = 0.5*(x_old[1:] + x_old[:-1])
        x = np.sort(np.concatenate([x_old, x_mid_points]))
        
        h = np.diff(x) # calculate new h
        n = len(x) 
        # initialize system of equations
        e, d, f, b = self._init_sle(x=x, h=h, eps=eps, n=n, advanced_solve=advanced_solve)
        u = thomas_algorithm(e=e, d=d, f=f, b=b) # solve

        return x, u


    def _init_sle(
        self,
        x: np.ndarray,
        h: np.ndarray,
        eps: float, 
        n: int, 
        advanced_solve: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # initialization
        e = np.zeros(n, np.float64)
        d = np.ones(n, np.float64)
        f = np.zeros(n, np.float64)

        # first & last row determined by u(x_0) = u(x_n-1) = 0
        if advanced_solve:
            b = np.zeros(n, np.float64)
            
            for i in range(1, n-1):
                # using advanced finite difference scheme
                mu_plus, mu_zero, mu_minus, nu_plus, nu_zero, nu_minus = self._get_adv_coefficients(i, x, h_plus=h[i], h_minus=h[i-1], eps=eps)

                b[i] = nu_plus * self.f(x[i+1]) + nu_zero * self.f(x[i]) + nu_minus * self.f(x[i-1])
                # NOTE: h shifted compared to problem description
                # i.e. h[0] = h_1 etc.
                if callable(self.c):
                    c_plus = self.c(x[i+1])
                    c_zero = self.c(x[i])
                    c_minus = self.c(x[i-1])
                else:
                    c_plus = self.c
                    c_zero = self.c
                    c_minus = self.c
                e[i] = -2 * eps**2 / ((h[i-1] + h[i]) * h[i-1]) + mu_minus * c_minus
                d[i] = 2 * eps**2 / (h[i-1] + h[i]) * (1 / h[i-1] + 1 / h[i]) + mu_zero * c_zero
                f[i] = -2 * eps**2 / ((h[i-1] + h[i]) * h[i]) + mu_plus * c_plus

        else:
            # basic finite difference method
            b = self.f(x)
            # setting boundary values
            b[0] = 0
            b[-1] = 0
            for i in range(1, n-1):
                # NOTE: h shifted compared to problem description
                # i.e. h[0] = h_1 etc.
                e[i] = -2 * eps**2 / ((h[i-1] + h[i]) * h[i-1])
                c = self.c(x[i]) if callable(self.c) else self.c
                d[i] = 2 * eps**2 / (h[i-1] + h[i]) * (1 / h[i-1] + 1 / h[i]) + c
                f[i] = -2 * eps**2 / ((h[i-1] + h[i]) * h[i])

        return e, d, f, b
    

    def _get_adv_coefficients(self, i, x, h_plus, h_minus, eps):
        # raise NotImplementedError

        # coefficients determined such that the finite difference scheme
        # yields exact solutions for all polynomials with degree <= 5
        if callable(self.c):
            c_plus = self.c(x[i+1])
            c_zero = self.c(x[i])
            c_minus = self.c(x[i-1])
        else:
            c_plus = self.c
            c_zero = self.c
            c_minus = self.c
        
        
        mu_plus = - 2 * eps**2 * (
            - 4 * h_minus**4 * h_plus**2 * c_plus + 3 * h_minus**3 * h_plus**3 * c_minus \
            - 3 * h_minus**3 * h_plus**3 * c_plus + 4 * h_minus**2 * h_plus**4 * c_minus \
            + 12 * eps**2 * h_minus**4 + 30 * eps**2 * h_minus**3 * h_plus \
            - 30 * eps**2 * h_minus * h_plus**3 - 12 * eps**2 * h_plus**4
        ) / (
            c_plus * h_plus * (
                - 3 * h_minus**4 * h_plus**3 * c_minus - 7 * h_minus**3 * h_plus**4 * c_minus \
                -4 * h_minus**2 * h_plus**5  * c_minus + 12 * eps**2 * h_minus**5 \
                + 42 * eps**2 * h_minus**4 * h_plus + 30 * eps**2 * h_minus**3 * h_plus**2 \
                + 30 * eps**2 * h_minus**2 * h_plus**3 + 42 * eps**2 * h_minus * h_plus**4 \
                + 12 * eps**2 * h_plus**5
            )
        )

        mu_zero = 2 * eps**2 * (
            8 * h_minus**4 * h_plus**2 * c_zero + 3 * h_minus**3 * h_plus**3 * c_minus \
            + 19 * h_minus**3 * h_plus**3 * c_zero + 4 * h_minus**2 * h_plus**4 * c_minus \
            + 8 * h_minus**2 * h_plus**4 * c_zero + 12 * eps**2 * h_minus**4 \
            + 6 * eps**2 * h_minus**3 * h_plus - 36 * eps**2 * h_minus**2 * h_plus**2 \
            + 6 * eps**2 * h_minus * h_plus**3 + 12 * eps**2 * h_plus**4
        ) / (
            c_zero * h_minus * h_plus * (
                - 3 * h_minus**3 * h_plus**3 * c_minus - 4 * h_minus**2 * h_plus**4 * c_minus \
                + 12 * eps**2 * h_minus**4 + 30 * eps**2 * h_minus**3 * h_plus \
                + 30 * eps**2 * h_minus * h_plus**3 + 12 * eps**2 * h_plus**4
            )
        )

        mu_minus = 12*eps**4 * (2*h_minus**3 + 3*h_minus**2 * h_plus - 3*h_minus * h_plus**2 - 2*h_plus**3) / (
            c_minus * h_minus * (
                - 3 * h_minus**3 * h_plus**3 * c_minus - 4 * h_minus**2 * h_plus**4 * c_minus \
                + 12 * eps**2 * h_minus**4 + 30 * eps**2 * h_minus**3 * h_plus \
                + 30 * eps**2 * h_minus * h_plus**3 + 12 * eps**2 * h_plus**4
            )
        )

        nu_plus = 2 * eps**2 * h_minus**3 * h_plus * (4 * h_minus + 3 * h_plus) / (
            - 3 * h_minus**4 * h_plus**3 * c_minus - 7 * h_minus**3 * h_plus**4 * c_minus \
            - 4 * h_minus**2 * h_plus**5 * c_minus + 12 * eps**2 * h_minus**5 \
            + 42 * eps**2 * h_minus**4 * h_plus + 30 * eps**2 * h_minus**3 * h_plus**2 \
            + 30 * eps**2 * h_minus**2 * h_plus**3 + 42 * eps**2 * h_minus * h_plus**4 \
            + 12 * eps**2 * h_plus**5
        )

        nu_zero = 2 * eps**2 * h_minus * h_plus * (8*h_minus**2 + 19*h_minus*h_plus + 8*h_plus**2) / (
            - 3 * h_minus**3 * h_plus**3 * c_minus - 4 * h_minus**2 * h_plus**4 * c_minus \
            + 12 * eps**2 * h_minus**4 + 30 * eps**2 * h_minus**3 * h_plus \
            + 30 * eps**2 * h_minus * h_plus**3 + 12 * eps**2 * h_plus**4
        )

        nu_minus = 2 * (3*h_minus + 4*h_plus) * eps**2 * h_minus * h_plus**3 / (
            - 3 * h_minus**4 * h_plus**3 * c_minus - 7 * h_minus**3 * h_plus**4 * c_minus \
            - 4 * h_minus**2 * h_plus**5 * c_minus + 12 * eps**2 * h_minus**5 \
            + 42 * eps**2 * h_minus**4 * h_plus + 30 * eps**2 * h_minus**3 * h_plus**2 \
            + 30 * eps**2 * h_minus**2 * h_plus**3 + 42 * eps**2 * h_minus * h_plus**4 \
            + 12 * eps**2 * h_plus**5
        )
        
        return mu_plus, mu_zero, mu_minus, nu_plus, nu_zero, nu_minus


    def _print_sle(
        self,
        n: int,
        e: np.ndarray,
        d: np.ndarray,
        f: np.ndarray,
        b: np.ndarray,
    ) -> None:
        print('Solving the system of linear equations given by the following augmented matrix (numbers rounded):')
        # fuse vectors into the respective augmented tridiagonal matrix
        sle = np.zeros((n, n+1))
        for i in range(n):
            if i > 0:
                sle[i, i-1] = e[i]
            if i < n:
                sle[i, i+1] = f[i]
            sle[i, i] = d[i]
            sle[i, -1] = b[i]

        sle = np.round(sle, 5) # round for nicer printing

        # print pretty table
        s = [[str(e) for e in row] for row in sle]
        for row in s:
            row.insert(-1, '|')
        lens = [max(map(len, col)) for col in zip(*s)]
        #fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        fmt = '  '.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))



class BoundedOptimizerStep():
    """random displacement with bounds"""
    def __init__(self, stepsize=0.5):
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds"""
        return np.clip(x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x)), 0., 1.)


    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """




    """Computes the crossentropy loss between the labels and predictions.
    Use this crossentropy loss function when there are two or more label
    classes. We expect labels to be provided in a `one_hot` representation. If
    you want to provide labels as integers, please use
    `SparseCategoricalCrossentropy` loss.  There should be `# classes` floating
    point values per feature.
    In the snippet below, there is `# classes` floating pointing values per
    example. The shape of both `y_pred` and `y_true` are
    `[batch_size, num_classes]`.
    Standalone usage:
    >>> y_true = [[0, 1, 0], [0, 0, 1]]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> cce = tf.keras.losses.CategoricalCrossentropy()
    >>> cce(y_true, y_pred).numpy()
    1.177
    >>> # Calling with 'sample_weight'.
    >>> cce(y_true, y_pred, sample_weight=tf.constant([0.3, 0.7])).numpy()
    0.814
    >>> # Using 'sum' reduction type.
    >>> cce = tf.keras.losses.CategoricalCrossentropy(
    ...     reduction=tf.keras.losses.Reduction.SUM)
    >>> cce(y_true, y_pred).numpy()
    2.354
    >>> # Using 'none' reduction type.
    >>> cce = tf.keras.losses.CategoricalCrossentropy(
    ...     reduction=tf.keras.losses.Reduction.NONE)
    >>> cce(y_true, y_pred).numpy()
    array([0.0513, 2.303], dtype=float32)
    Usage with the `compile()` API:
    ```python
    model.compile(optimizer='sgd',
                  loss=tf.keras.losses.CategoricalCrossentropy())
    ```
    """

"""
Args:
      apply_class_balancing: A bool, whether to apply weight balancing on the
        binary classes 0 and 1.
      alpha: A weight balancing factor for class 1, default is `0.25` as
        mentioned in reference [Lin et al., 2018](
        https://arxiv.org/pdf/1708.02002.pdf).  The weight for class 0 is
        `1.0 - alpha`.
      gamma: A focusing parameter used to compute the focal factor, default is
        `2.0` as mentioned in the reference
        [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf).
      from_logits: Whether to interpret `y_pred` as a tensor of
        [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
        assume that `y_pred` are probabilities (i.e., values in `[0, 1]`).
      label_smoothing: Float in `[0, 1]`. When `0`, no smoothing occurs. When >
        `0`, we compute the loss between the predicted labels and a smoothed
        version of the true labels, where the smoothing squeezes the labels
        towards `0.5`. Larger values of `label_smoothing` correspond to heavier
        smoothing.
      axis: The axis along which to compute crossentropy (the features axis).
        Defaults to `-1`.
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras`, `compile()` and `fit()`, using `SUM_OVER_BATCH_SIZE` or
        `AUTO` will raise an error. Please see this custom training [tutorial](
        https://www.tensorflow.org/tutorials/distribute/custom_training) for
        more details.
      name: Name for the op. Defaults to 'binary_focal_crossentropy'.

Returns:
    A `tf.Tensor`. Has the same type as `x`.
    """