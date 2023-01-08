import numpy as np
from typing import List, Optional, Tuple, Callable, Union
from scipy.optimize import basinhopping
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
import math


############# TODO #############
# TODO check format of multi-plotters --> modify tight layout rect
# NOTE: multiple solvers do not support reference solution !


def thomas_algorithm(
    e: np.ndarray,
    d: np.ndarray,
    f: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Thomas algorithm for solving tridiagonal systems of equations.

    Args:
        e: A vector of the entries below the diagonal.
        d: A vector of the entries on the diagonal.
        f: A vector of the entries above the diagonal.
        b: A vector of the constants of the system of equations.

    Returns:
        A solution to the tridiagonal system of equations.
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
    Class instances represent specific reaction diffusion equations
    of the form \$\\varepsilon^2 u''(x) + c(x)*u(x) = f(x)\$.
    The primary class method is `solve` which will compute an 
    numerical approximation of the solution. In addition, several methods
    are exposed for convenience. These enable the computation of 
    multiple approximations for various values of epsilon or numbers
    of mesh points at once as well as quick plotting.

    Args:
        f: A callable representing the function on the right side of the 
            reaction diffusion equation.
        c: The coefficient of the reaction diffusion equation, can be 
            constant or a callable.
        gamma_iterations: Maximum number of iterations for finding the 
            global minimum of c.
        gamma_start: Initial value for finding the 
            global minimum of c.
        gamma_init_step: Initial step size for finding the 
            global minimum of c.
    """

    def __init__(
        self,
        f: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
        c: Union[int, float, Callable[[Union[float, np.ndarray]], Union[int, float, np.ndarray]]],
        gamma_iterations: int = 100,
        gamma_start: float = 0.5,
        gamma_init_step: float = 0.5,
    ):
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
        Computes an approximation of the solution u(x) of the reaction
        diffusion problem using a uniform or Shishkin mesh and finite 
        difference methods.

        Args:
            eps: A value for epsilon in the reaction diffusion equation.
            n: The number of mesh points in the discretization.
            shishkin_mesh: A boolean, whether to use a piecewise-uniform
                Shishkin mesh rather than a uniform mesh (default).
            sigma: Parameter of the Shishkin mesh, only relevant if `shishkin_mesh` 
                is `True`. Only values greater equal 2 are accepted or greater
                equal 4 when `advanced_solve` is `True`, defaults to `2`.
            advanced_solve: A boolean, whether to use an advanced finite 
                difference scheme to determine the tridiagonal system of equations.
                The default is `False`.
            double_mesh: A boolean, whether to compute a reference solution  
                using the double mesh principle. If this is set to `True`, an additional
                tuple containing the x- and y-values of this reference solution
                is returned. The default is `False`.
            verbose: A boolean which controls the verbosity. If set to `True`,
                the system of equations and further information will be printed.
                The default is `False`.

        Returns:
            A tuple of the x- and y-values of the approximation at the mesh points
            if `double_mesh` is `False` or otherwise this tuple and an additional
            tuple containing the x- and y-values of the reference solution.
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
        mesh: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Plot an approximation of u(x).

        Args:
            eps: A value for epsilon in the reaction diffusion equation.
            n: The number of mesh points in the discretization.
            shishkin_mesh: A boolean, whether to use a piecewise-uniform
                Shishkin mesh rather than a uniform mesh (default).
            sigma: Parameter of the Shishkin mesh, only relevant if `shishkin_mesh` 
                is `True`. Only values greater equal 2 are accepted or greater
                equal 4 when `advanced_solve` is `True`, defaults to `2`.
            advanced_solve: A boolean, whether to use an advanced finite 
                difference scheme to determine the tridiagonal system of equations.
                The default is `False`.
            verbose: A boolean which controls the verbosity. If set to `True`,
                the system of equations and further information will be printed.
                The default is `False`.
            interpolation: A string defining the method to interpolate the 
                approximations at the mesh points. The available options are
                `'linear'`, `'quadratic'` and `'cubic'`. Anything else will 
                result in no interpolation (default).
            scatter: A boolean, whether to draw points for the approximations
                at the mesh points. The default is `True`.
            mesh: A boolean, whether to draw vertical lines at the mesh points.
                These will be drawn instead of the standard grid lines. The 
                default is `False`.

        Returns:
            A tuple of the x- and y-values of the approximation at the mesh points.
        """
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

        if mesh:
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
    ) -> List:
        """
        Calculate the approximations of u(x) for various values of epsilon.

        Args:
            eps: A list of values for epsilon in the reaction diffusion equation.
            n: The number of mesh points in the discretization.
            shishkin_mesh: A boolean, whether to use a piecewise-uniform
                Shishkin mesh rather than a uniform mesh (default).
            sigma: Parameter of the Shishkin mesh, only relevant if `shishkin_mesh` 
                is `True`. Only values greater equal 2 are accepted or greater
                equal 4 when `advanced_solve` is `True`, defaults to `2`.
            advanced_solve: A boolean, whether to use an advanced finite 
                difference scheme to determine the tridiagonal system of equations.
                The default is `False`.
            verbose: A boolean which controls the verbosity. If set to `True`,
                the system of equations and further information will be printed.
                The default is `False`.

        Returns:
            A list of tuples of the x- and y-values of the approximation at the 
            mesh points for each given epsilon value.
        """
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
        mesh: bool = False,
    ) -> List:
        """
        Plot the approximations of u(x) for various values of epsilon.

        Args:
            eps: A list of values for epsilon in the reaction diffusion equation.
            n: The number of mesh points in the discretization.
            shishkin_mesh: A boolean, whether to use a piecewise-uniform
                Shishkin mesh rather than a uniform mesh (default).
            sigma: Parameter of the Shishkin mesh, only relevant if `shishkin_mesh` 
                is `True`. Only values greater equal 2 are accepted or greater
                equal 4 when `advanced_solve` is `True`, defaults to `2`.
            advanced_solve: A boolean, whether to use an advanced finite 
                difference scheme to determine the tridiagonal system of equations.
                The default is `False`.
            verbose: A boolean which controls the verbosity. If set to `True`,
                the system of equations and further information will be printed.
                The default is `False`.
            interpolation: A string defining the method to interpolate the 
                approximations at the mesh points. The available options are
                `'linear'`, `'quadratic'` and `'cubic'`. Anything else will 
                result in no interpolation (default).
            subplots: A boolean, whether to create a subplot for each approximation
                if set to `True` or to draw all approximations in one plot. 
                The latter may increase comparability across approximations but 
                gets chaotic when using to many approximations at once.
                The default is `True`.
            scatter: A boolean, whether to draw points for the approximations
                at the mesh points. The default is `True`.
            mesh: A boolean, whether to draw vertical lines at the mesh points.
                These will be drawn instead of the standard grid lines. The 
                default is `False`.

        Returns:
            A list of tuples of the x- and y-values of the approximation at the 
            mesh points for each given epsilon value.
        """
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

                if mesh:
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

            if mesh and not shishkin_mesh:
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

        return solutions


    def solve_multiple_n(
        self,
        n: Union[List, np.ndarray],
        eps: float = 1,
        shishkin_mesh: bool = False,
        sigma: float = 2,
        advanced_solve: bool = False,
        verbose: bool = False,
    ) -> List:
        """
        Calculate the approximations of u(x) for various numbers of mesh points.

        Args:
            n: A list of values for the number of mesh points in the 
                discretization.
            eps: A value for epsilon in the reaction diffusion equation.
            shishkin_mesh: A boolean, whether to use a piecewise-uniform
                Shishkin mesh rather than a uniform mesh (default).
            sigma: Parameter of the Shishkin mesh, only relevant if `shishkin_mesh` 
                is `True`. Only values greater equal 2 are accepted or greater
                equal 4 when `advanced_solve` is `True`, defaults to `2`.
            advanced_solve: A boolean, whether to use an advanced finite 
                difference scheme to determine the tridiagonal system of equations.
                The default is `False`.
            verbose: A boolean which controls the verbosity. If set to `True`,
                the system of equations and further information will be printed.
                The default is `False`.

        Returns:
            A list of tuples of the x- and y-values of the approximation at the 
            mesh points for each given value of n.
        """
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
        """
        Plots the approximations of u(x) for various numbers of mesh points.

        Args:
            n: A list of values for the number of mesh points in the 
                discretization.
            eps: A value for epsilon in the reaction diffusion equation.
            shishkin_mesh: A boolean, whether to use a piecewise-uniform
                Shishkin mesh rather than a uniform mesh (default).
            sigma: Parameter of the Shishkin mesh, only relevant if `shishkin_mesh` 
                is `True`. Only values greater equal 2 are accepted or greater
                equal 4 when `advanced_solve` is `True`, defaults to `2`.
            advanced_solve: A boolean, whether to use an advanced finite 
                difference scheme to determine the tridiagonal system of equations.
                The default is `False`.
            verbose: A boolean which controls the verbosity. If set to `True`,
                the system of equations and further information will be printed.
                The default is `False`.
            interpolation: A string defining the method to interpolate the 
                approximations at the mesh points. The available options are
                `'linear'`, `'quadratic'` and `'cubic'`. Anything else will 
                result in no interpolation (default).
            subplots: A boolean, whether to create a subplot for each approximation
                if set to `True` or to draw all approximations in one plot. 
                The latter may increase comparability across approximations but 
                gets chaotic when using to many approximations at once.
                The default is `True`.
            scatter: A boolean, whether to draw points for the approximations
                at the mesh points. The default is `True`.

        Returns:
            A list of tuples of the x- and y-values of the approximation at the 
            mesh points for each given value of n.
        """
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
        """
        Plots the approximations using the simple and the advanced 
        finite difference methods with uniform and Shishkin meshes
        respectively. 

        Args:
            n: The number of mesh points in the discretization.
            eps: A value for epsilon in the reaction diffusion equation.
            sigma: A list of length 3 containing a parameter of the Shishkin 
                mesh for each method. The first value must be greater equal 
                than `2`, the other two greater equal than `4`. Defaults 
                to `[2, 4, 4]`.
            interpolation: A string defining the method to interpolate the 
                approximations at the mesh points. The available options are
                `'linear'`, `'quadratic'` and `'cubic'`. Anything else will 
                result in no interpolation (default).
            subplots: A boolean, whether to create a subplot for each approximation
                if set to `True` or to draw all approximations in one plot. 
                The latter may increase comparability across approximations but 
                gets chaotic when using to many approximations at once.
                The default is `True`.
            scatter: A boolean, whether to draw points for the approximations
                at the mesh points. The default is `True`.
        """
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

            fig.suptitle(f'Approximations of u(x) with eps = {eps} and {n} mesh points \
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
            plt.title(f'Approximations of u(x) with eps = {eps} and {n} mesh points \
using different approaches', fontsize=20)
            plt.grid()
            plt.show()


    def _get_gamma(self):
        """
        Numerically approximates the global minimum of the coefficient c
        in the reaction diffusion equation.

        Returns:
            The square root of the global minimum of the coefficient c
            in the reaction diffusion equation.
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
        Generates a reference solution to a given approximation using the 
        double mesh principle.

        Args:
            x_old: An array containing the mesh points of the approximation
                for which a reference solution shall be calculated.
            eps: A value for epsilon in the reaction diffusion equation.
            n: The number of mesh points in the discretization.
            advanced_solve: A boolean, whether to use an advanced finite 
                difference scheme to determine the tridiagonal system of equations.
                The default is `False`.

        Returns:
            A tuple of the x- and y-values of the reference solution.
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
        """
        Creates the tridiagonal system of linear equations via finite difference methods.

        Args:
            x: An array of the mesh points.
            h: An array of the local mesh sizes.
            eps: A value for epsilon in the reaction diffusion equation.
            n: The number of mesh points in the discretization.
            advanced_solve: A boolean, whether to use an advanced finite 
                difference scheme to determine the tridiagonal system of equations.
                The default is `False`.

        Returns:
            A tuple of four arrays. In order: The coefficients below, on and above
            the diagonal and the constants of the system of equations.
        """
        # initialization
        e = np.zeros(n, np.float64)
        d = np.ones(n, np.float64)
        f = np.zeros(n, np.float64)

        # first & last row determined by u(x_0) = u(x_n-1) = 0
        if advanced_solve:
            b = np.zeros(n, np.float64)
            
            for i in range(1, n-1):
                # using advanced finite difference scheme
                mu_plus, mu_zero, mu_minus, nu_plus, nu_zero, nu_minus = self._get_adv_coefficients(h_plus=h[i], h_minus=h[i-1])
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
                d[i] = 2 * eps**2 / (h[i-1] + h[i]) * (1 / h[i] + 1 / h[i-1]) + mu_zero * c_zero
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
    

    def _get_adv_coefficients(self, h_plus, h_minus):
        """
        Compute the coefficients in the advanced finite difference scheme. 
        These are determined such that the scheme is exact for all 
        polynomials with a maximum degree of 4.

        Args:
            h_plus: x[i+1] - x[i].
            h_plus: x[i] - x[i-1].

        Returns:
            The six coefficients needed in the advanced finite difference scheme.
            In order these are `mu_plus`, `mu_zero`, `mu_minus`, `nu_plus`, 
            `nu_zero` and `nu_minus`.
        """
        nu_plus = (1 - h_minus**2/(h_plus * (h_plus + h_minus))) / 6
        nu_zero = (4 + h_minus**2/(h_plus * (h_plus + h_minus)) + h_plus**2/(h_minus * (h_plus + h_minus))) / 6
        nu_minus = (1 - h_plus**2/(h_minus * (h_plus + h_minus))) / 6

        mu_plus = (1 - h_minus**2/(h_plus * (h_plus + h_minus))) / 6
        mu_zero = (4 + h_minus**2/(h_plus * (h_plus + h_minus)) + h_plus**2/(h_minus * (h_plus + h_minus))) / 6
        mu_minus = (1 - h_plus**2/(h_minus * (h_plus + h_minus))) / 6
        
        return mu_plus, mu_zero, mu_minus, nu_plus, nu_zero, nu_minus


    def _print_sle(
        self,
        n: int,
        e: np.ndarray,
        d: np.ndarray,
        f: np.ndarray,
        b: np.ndarray,
    ) -> None:
        """
        Utility function to print out the augmented matrix of 
        the tridiagonal system of equations.

        Args:
            n: The number of mesh points.
            e: A vector of the entries below the diagonal.
            d: A vector of the entries on the diagonal.
            f: A vector of the entries above the diagonal.
            b: A vector of the constants of the system of equations.
        """
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
    """
    Modified optimizer step to ensure that the optimizer does not
    step out of bounds. `__call__` will return a random step within
    the bounds.
    
    Args:
        stepsize: The initial stepsize of the optimizer.
    """
    def __init__(self, stepsize=0.5):
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds"""
        return np.clip(x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x)), 0., 1.)