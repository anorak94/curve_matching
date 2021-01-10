import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

class Coordinates(object):
    """
    Container for grid coordinates.
    Attributes
    ----------
    domain : nd-array
        Domain of the coordinate system.
    tensor : nd-array
        Grid coordinates.
    homogenous : nd-array
        `Homogenous` coordinate system representation of grid coordinates.
    """

    def __init__(self, domain, spacing=None):

        self.spacing = 1.0 if not spacing else spacing
        self.domain = domain
        self.xrange = jnp.arange(0., domain[1], 1)
        self.yrange = jnp.arange(0., domain[3], 1)
        self.tensor = jnp.meshgrid(self.xrange, self.yrange, indexing = "xy")

class ThinPlateSpline():

    MODEL='Thin Plate Spline (TPS)'

    DESCRIPTION="""
        Computes a thin-plate-spline deformation model, as described in:
        Bookstein, F. L. (1989). Principal warps: thin-plate splines and the
        decomposition of deformations. IEEE Transactions on Pattern Analysis
        and Machine Intelligence, 11(6), 567-585.
        """

    def __init__(self, coordinates, p0, p1, fix, mov):
        
        self.full_basis = None 
        self.coordinates = coordinates
        self.parameters_analytical = None
        self.red_basis = None
        self.p0  = p0
        self.p1  = p1
        self.im1 = fix
        self.im2 = mov
        if self.full_basis == None:
            self.__basis()
        if self.red_basis == None:
            self.__basis_red()
        #Model.__init__(self, coordinates)
    

    def U(self, r):
        """
        Kernel function, applied to solve the biharmonic equation.
        Parameters
        ----------
        r: float
            Distance between sample and coordinate point.
        Returns
        -------
        U: float
           Evaluated kernel.
        """

        return jnp.multiply(-jnp.power(r,2), jnp.log(jnp.power(r,2) + 1e-20))

    def spline_error(self, parameters):
        _p0, _p1, _projP0, error = self.__splineError( parameters)
        return error 
    
    def image_similarity_error(self, parameters):
        warp_field = self.transform(parameters)
        im2_warp = jax.scipy.ndimage.map_coordinates(self.im2, warp_field, order = 1).T
        f1 = jax.scipy.signal.correlate2d(self.im1, im2_warp, mode='full', boundary='fill', fillvalue=0, precision=None)
        f2 = jax.scipy.signal.correlate2d(self.im1, self.im1, mode='full', boundary='fill', fillvalue=0, precision=None)
        f3 = jax.scipy.signal.correlate2d(im2_warp, im2_warp, mode='full', boundary='fill', fillvalue=0, precision=None)
        return jnp.mean(f1)/(jnp.mean(f2)*jnp.mean(f3))
        
    def total_error (self, alpha_spline, alpha_image, parameters):
        spline_error = self.spline_error(parameters)
        image_error = self.image_similarity_error (parameters)
        #print (f"Spline error is {spline_error}, Image error is {image_error}")
        return alpha_spline*spline_error + alpha_image*image_error
        
    
    def fit(self, lmatrix):
        """
        Estimates the best fit parameters that define a warp field, which
        deforms feature points p0 to p1.
        Parameters
        ----------
        p0: nd-array
            Image features (points).
        p1: nd-array
            Template features (points).
        lmatrix: boolean
            Enables the spline design matrix when returning.
        Returns
        -------
        parameters: nd-array
            Model parameters.
        error: float
            Sum of RMS error between p1 and alinged p0.
        L: nd-array
            Spline design matrix, optional (using lmatrix keyword).
        """

        K =  self.U(jnp.linalg.norm(self.p0[:, None, :] - self.p0[None, :, :], axis=-1))
        
        P = jnp.hstack((jnp.ones((self.p0.shape[0], 1)), self.p0))

        L = jnp.vstack((jnp.hstack((K,P)),
                       jnp.hstack((P.transpose(), jnp.zeros((3,3))))))

        Y = jnp.vstack( (self.p1, jnp.zeros((3, 2))) )

        parameters = jnp.dot(jnp.linalg.inv(L), Y)
        
        self.parameters_analytical = parameters

        if lmatrix:
            return parameters, L
        else:
            return parameters

    
    def __splineError(self, parameters):
        """
        Estimates the point alignment and computes the alignment error.
        Parameters
        ----------
        p0: nd-array
            Image features (points).
        p1: nd-array
            Template features (points).
        parameters: nd-array
            Thin-plate spline parameters.
        Returns
        -------
        error: float
            Alignment error between p1 and projected p0 (RMS).
        """

        # like __basis, compute a reduced set of basis vectors.

        # compute the alignment error.
        projP0 = jnp.vstack( [
           jnp.dot(self.red_basis, parameters[:,1]),
           jnp.dot(self.red_basis, parameters[:,0])
           ]
           ).T
        error = jnp.sqrt(
           (projP0[:,0] - self.p1[:,0])**2 + (projP0[:,1] - self.p1[:,1])**2
           ).sum()

        return self.p0, self.p1, projP0, error

    def __basis(self):
        """
        Forms the thin plate spline deformation basis, which is composed of
        a linear and non-linear component.
        Parameters
        ----------
        p0: nd-array
            Image features (points).
        """
        
        
        self.full_basis = jnp.zeros((self.coordinates.tensor[0].size, len(self.p0)+3))
        # nonlinear, spline component.
        for index, p in enumerate( self.p0 ):
            self.full_basis = jax.ops.index_update(self.full_basis, jax.ops.index[:, index], self.U(
                jnp.sqrt(
                    (p[0]-self.coordinates.tensor[1])**2 +
                    (p[1]-self.coordinates.tensor[0])**2
                    )
            ).flatten())
            

        # linear, affine component
        self.full_basis = jax.ops.index_update(self.full_basis, jax.ops.index[:, -3], 1.)
        self.full_basis = jax.ops.index_update(self.full_basis, jax.ops.index[:, -2], self.coordinates.tensor[1].flatten())
        self.full_basis = jax.ops.index_update(self.full_basis, jax.ops.index[:, -1], self.coordinates.tensor[0].flatten())
        
    def __basis_red(self):
        """
        Forms the thin plate spline deformation basis, which is composed of
        a linear and non-linear component.
        Parameters
        ----------
        p0: nd-array
            Image features (points).
        """
        
        
        self.red_basis = jnp.zeros((self.p0.shape[0], len(self.p0)+3))
        # nonlinear, spline component.
        for index, p in enumerate( self.p0 ):
            self.red_basis = jax.ops.index_update(self.red_basis, jax.ops.index[:, index], self.U(
                jnp.sqrt(
                    (p[0]-self.p1[:,0])**2 +
                    (p[1]-self.p1[:,1])**2
                    )
            ).flatten())
            

        # linear, affine component
        self.red_basis = jax.ops.index_update(self.red_basis, jax.ops.index[:, -3], 1.)
        self.red_basis = jax.ops.index_update(self.red_basis, jax.ops.index[:, -2], self.p1[:,1])
        self.red_basis = jax.ops.index_update(self.red_basis, jax.ops.index[:, -1], self.p1[:,0])
        

    def transform(self, parameters):
        """
        A "thin-plate-spline" transformation of coordinates.
        Parameters
        ----------
        parameters: nd-array
            Model parameters.
        Returns
        -------
        coords: nd-array
           Deformation coordinates.
        """

        shape = self.coordinates.tensor[0].shape
        return jnp.array( [ jnp.dot(self.full_basis, parameters[:,1]).reshape(shape),
                           jnp.dot(self.full_basis, parameters[:,0]).reshape(shape)
                         ]
                       )

    def warp(self, parameters):
        """
        Computes the warp field given model parameters.
        Parameters
        ----------
        parameters: nd-array
            Model parameters.
        Returns
        -------
        warp: nd-array
           Deformation field.
        """

        return self.transform(parameters)
