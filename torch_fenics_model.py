import torch
import torch_fenics

from fenics import *
from fenics_adjoint import *
from ufl import nabla_div

class HomogeneousBeam(torch_fenics.FEniCSModule):
  # TODO: how to make tolerance a variable (i.e. defining the tol in BC without running into error)

    # Construct variables which can be reused in the constructor
    def __init__(self):
        # Call super constructor
        super().__init__()

        # Scaled variables
        self.bc = None
        self.L = 2
        self.W = 0.2

        self.rho = 1
        self.delta = self.W/self.L
        self.gamma = 0.4*self.delta**2
        self.g = self.gamma
        self.tol = 1e-14 # boundary condition

        # Create function space                                        # of cells in x-y-z
        self.mesh = BoxMesh(Point(0, 0, 0), Point(self.L, self.W, self.W), 10, 3, 3)
        self.V = VectorFunctionSpace(self.mesh, 'CG', 1)

        # Create trial and test functions
        self.v = TestFunction(self.V)
        self.f = Constant((500, 0, -self.rho*self.g*1000000)) # distributed load pushin on z-dim
        self.T = Constant((0, 0, 0)) # traction stress; force vector; where does it apply?


    # Define strain and stress
    def epsilon(self, u):
        return 0.5*(nabla_grad(u) + nabla_grad(u).T)

    def sigma(self, u):
        return self.lambda_*nabla_div(u)*Identity(self.d) + 2*self.mu*self.epsilon(u)


    # BC's condition excerpted from https://fenicsproject.org/qa/13149/clamped-beam-under-the-end-load-problem/
    @staticmethod
    def clamped_boundary_left(x, on_boundary):
        # tolerance is manually defined here as 1e-14
        return on_boundary and x[0] < 1e-14 # should be using self.tol
    @staticmethod
    def clamped_boundary_right(x, on_boundary):
        # tolerance is manually defined here as 1e-14
        return on_boundary and (x[0] - 1) < 1e-14 # should be using self.tol




    # def solve(self, g, mu, beta):
    def solve(self, mu, beta):
        # Parameters to be optimized
        self.mu = mu
        self.beta = beta
        self.lambda_ = self.beta


        self.u = TrialFunction(self.V)
        self.d = self.u.geometric_dimension()  # space dimension
        self.a = inner(self.sigma(self.u), self.epsilon(self.v))*dx

        
        L = dot(self.f, self.v)*dx + dot(self.T, self.v)*ds

        # Construct boundary condition
                                          # the x-y-z location on the bc
        self.bc_l = DirichletBC(self.V, Constant((0, 0, 0)), self.clamped_boundary_left)
        self.bc_r = DirichletBC(self.V, Constant((0, 0, 0)), self.clamped_boundary_right)


        # Solve
        self.u = Function(self.V) # displacement
        solve(self.a == L, self.u, [self.bc_l, self.bc_r])
  
        # Return the solution
        return self.u

    def input_templates(self):
        # Declare input shapes
        # return Constant((0, 0, 0)), 
        return Constant(1), Constant(1)

    def exec(self, expression):
        # To work with fenics' functions
        if 'return' in expression: return exec(expression.split(' ')[1])
        else: exec(expression)
