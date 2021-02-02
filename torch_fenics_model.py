import torch
import torch_fenics

from fenics import *
from fenics_adjoint import *
from ufl import nabla_div

class Corroded(SubDomain):
  def inside(self, x, on_boundary):
    tol = 1E-14
    return on_boundary and (between(x[0], (0.5, 0.7)) )


class Healthy(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and (between(x[0], (0., 0.5)) or between(x[0], (0.7, 1.)))

class HomogeneousBeam(torch_fenics.FEniCSModule):
    # Construct variables which can be reused in the constructor
    def __init__(self):
        # Call super constructor
        super().__init__()
        #TODO: make L, W variables
        # Scaled variables
        self.bc = None
        self.L = 1
        self.W = 0.2

        self.rho = 1
        self.delta = self.W/self.L
        self.gamma = 0.4*self.delta**2
        self.g = self.gamma
        self.tol = 1e-14 # boundary condition

        # Create function space                                        # of cells in x-y-z
        self.mesh = BoxMesh(Point(0, 0, 0), Point(self.L, self.W, self.W), 10, 3, 3)
        self.V = VectorFunctionSpace(self.mesh, 'P', 1)

        # Create trial and test functions
        self.v = TestFunction(self.V)
        self.T = Constant((0, 0, 0)) 


    # Define strain and stress
    def epsilon(self, u):
        return 0.5*(nabla_grad(u) + nabla_grad(u).T)

    def sigma(self, u):
        return self.lambda_*nabla_div(u)*Identity(self.d) + 2*self.mu*self.epsilon(u)


    # BC's condition excerpted from https://fenicsproject.org/qa/13149/clamped-beam-under-the-end-load-problem/
    @staticmethod
    def clamped_boundary_left(x, on_boundary):
        tol = 1E-14 # tolerance is manually defined here as 1e-14
        return on_boundary and near(x[0], 0., tol) # should be using self.tol
    
    @staticmethod
    def clamped_boundary_right(x, on_boundary):
        tol = 1E-14# tolerance is manually defined here as 1e-14
        return on_boundary and near(x[0], 1., tol) # should be using self.tol


    def solve(self, mu, beta, force_multiplier):
        # Parameters to be optimized
        self.mu = mu
        self.beta = beta
        self.lambda_ = self.beta
        self.f = Constant((0, 0, -self.rho*self.g*force_multiplier))


        self.u = TrialFunction(self.V)
        self.d = self.u.geometric_dimension()  # space dimension
        self.a = inner(self.sigma(self.u), self.epsilon(self.v))*dx

        
        L = dot(self.f, self.v)*dx + dot(self.T, self.v)*ds

        # Construct boundary condition
        self.bc_l = DirichletBC(self.V, Constant((0, 0, 0)), self.clamped_boundary_left)
 


        # Solve
        self.u = Function(self.V) # displacement
        solve(self.a == L, self.u, self.bc_l)
        # plot(self.u)


        # Return the solution
        return self.u

    def input_templates(self):
        # Declare input shapess
        return Constant((1)), Constant((1)), Constant((1))

    def exec(self, expression):
        # To work with fenics' functions
        if 'return' in expression: return exec(expression.split(' ')[1])
        else: exec(expression)