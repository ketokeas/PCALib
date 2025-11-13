import copy
import time
from functools import partial

from tqdm import tqdm

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from jax import jit
import numpy as np
from jax.numpy.linalg import inv, pinv
from . import utils

class Point:
    _FIELDS = (
        "W", "v", "L", "r", "R",
        "tilde_W", "U", "tilde_v",
        "S", "p", "tilde_R"
    )
    def __init__(self, W, v, L, r, R, tilde_W, U, tilde_v, S, p, tilde_R):
        """
        Initialize a Point in the optimization space.

        Parameters:
        - W: JAX array of size (K, K).
        - v: JAX array of size (K, K, K, K).
        - L: JAX array of size (K, K).
        - r: JAX array of size (K, K, K, K).
        - R: JAX array of size (D, K, K).
        - tilde_W: JAX array of size (K, K).
        - U: JAX array of size (K, K).
        - tilde_v: JAX array of size (K, K, K, K).
        - S: JAX array of size (K, K).
        - p: JAX array of size (K, K, K, K).
        - tilde_R: JAX array of size (D, K, K).
        """
        self.W, self.v, self.L, self.r, self.R = W, v, L, r, R
        self.tilde_W, self.U, self.tilde_v = tilde_W, U, tilde_v
        self.S, self.p, self.tilde_R = S, p, tilde_R

    # Iterator over attributes to later flatten/unflatten them to do the high-dim optimization
    def _iter_fields(self):
        for name in self._FIELDS:
            yield getattr(self, name)

    def tree_flatten(self):
        return tuple(self._iter_fields()), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def as_flat_vector(self) -> jnp.ndarray:
        """Flatten all variables into a single vector for optimization."""
        return jnp.concatenate([x.ravel() for x in self._iter_fields()])

    def update_from_vector(self, flat_vector):
        """Update the attributes of the Point instance from a flat vector using index_map."""
        idx_map = self.index_map()  # Get slices for all attributes

        for key, idx_slice in idx_map.items():
            setattr(self, key, flat_vector[idx_slice].reshape(getattr(self, key).shape))

    def index_map(self):
        """
        Returns a dictionary mapping variable names to their index slices
        in the flattened vector, generated dynamically from _FIELDS.
        """
        idx_map = {}
        start = 0
        for name in self._FIELDS:
            arr = getattr(self, name)
            size = arr.size
            idx_map[name] = slice(start, start + size)
            start += size
        return idx_map



class Potential:
    def __init__(self, bar_sigma, bar_e, G, bar_xi, Z, Delta, bar_x, Xi):
        """
        Initialize the Potential_old with individual parameters.

        Parameters:
        - bar_sigma: Array of size N.
        - bar_e: Array of size N x K.
        - G: Array of size D x N x N.
        - bar_xi: Array of size D x K.
        - Z: Matrix of size T x T.
        - Delta: Matrix of size T x T.
        - bar_x: Array of size T x K.
        - Xi: Array of size T x T.
        """
        N = jnp.shape(bar_sigma)[0]

        self.bar_sigma, self.bar_e, self.G = jnp.array(bar_sigma), jnp.array(bar_e), jnp.array(G)
        self.bar_xi, self.Z, self.Delta = jnp.array(bar_xi), jnp.array(Z),  jnp.array(Delta)
        self.bar_x, self.Xi  = jnp.array(bar_x), jnp.array(Xi)

        T = self.Z.shape[0]
        circulant_matrix = jnp.array([jnp.roll(self.bar_x, t, axis=0) for t in range(T)])
        self.X = jnp.einsum("tsk,sr,url->kltu", circulant_matrix, self.Xi, circulant_matrix)


    @classmethod
    def from_npz(cls, filename):
        """
        Load the Potential from a .npz file.

        Parameters:
        - filename: Path to the .npz file.

        Returns:
        - Instance of Potential class.
        """
        data = np.load(filename)
        return cls(
            data["bar_sigma"],
            data["bar_e"],
            data["G"],
            data["bar_xi"],
            data["Z"],
            data["Delta"],
            data["bar_x"],
            data["Xi"]
        )


    def save_as_npz(self, filename):
        """
        Save the Potential_old instance to a .npz file.

        Parameters:
        - filename: Path to the .npz file.
        """
        np.savez(filename,
                 bar_sigma=self.bar_sigma,
                 bar_e=self.bar_e,
                 G=self.G,
                 bar_xi=self.bar_xi,
                 Z=self.Z,
                 Delta=self.Delta,
                 bar_x=self.bar_x,
                 Xi=self.Xi
                 )

    @jit
    def update_X(self):
        T = self.Z.shape[0]
        circulant_matrix = jnp.array([jnp.roll(self.bar_x, t, axis=0) for t in range(T)])
        self.X = jnp.einsum("tsk,sr,url->kltu", circulant_matrix, self.Xi, circulant_matrix)

    def tree_flatten(self):
        children = (self.bar_sigma, self.bar_e, self.G, self.bar_xi, self.Z, self.Delta, self.bar_x, self.Xi, self.X)
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        bar_sigma, bar_e, G, bar_xi, Z, Delta, bar_x, Xi, _ = children
        return cls(bar_sigma, bar_e, G, bar_xi, Z, Delta, bar_x, Xi)

    @staticmethod
    @jit
    def tensor_contraction_with_e(tensor, bar_e):
        """
        Perform the contraction (tensor \cdot \bar{e} \bar{e}^T).
        """
        return jnp.einsum("klmn,ik,il->imn", tensor, bar_e, bar_e)

    @staticmethod
    @jit
    def tensor_contraction_with_X(tensor, X):
        """
        Perform the contraction (tensor \cdot \mathcal{X}).
        """
        return jnp.tensordot(tensor, X, axes=([0, 1], [0, 1]))

    @staticmethod
    @jit
    def tensor_dot(tensor1, tensor2):
        """
        Perform the contraction (tensor1 \circ tensor2).
        """
        return jnp.sum(tensor1 * tensor2)

    @jit
    def free_energy(self, point):
        """
        Compute the free energy value at a given point.
        """
        sp = self  # Static parameters
        p = point  # Point_old instance (dynamic variables)

        N = sp.bar_sigma.shape[0]
        T = sp.Z.shape[0]
        K = p.W.shape[0]
        Delta_no_J = sp.Delta @ (jnp.eye(T) - jnp.ones((T, T)) / T)  # Δ(I - J_T/T)
        tilde_Z = sp.Z @ (jnp.eye(T) - jnp.ones((T, T)) / T)

        # First term
        A_i = self.tensor_contraction_with_e(p.v, sp.bar_e)
        A_i += jnp.einsum("i,kl->ikl", sp.bar_sigma ** 2, p.tilde_W)
        A_i += jnp.einsum("i,kl->ikl", jnp.ones(N), p.U)

        B_i = jnp.einsum("i,kl->ikl", sp.bar_sigma ** 2, p.S)
        B_i += self.tensor_contraction_with_e(p.p, sp.bar_e)
        tRe = jnp.einsum("dkl,il->idk", p.tilde_R, sp.bar_e)
        tReetR = jnp.einsum("idk,idl->idkl", tRe, tRe)
        B_i -= jnp.einsum("dii,idkl->ikl", sp.G, tReetR)

        trace_function = lambda matA, matB: jnp.trace(jnp.linalg.inv(matA) @ matB) / (2 * N)
        first_term = - jnp.sum(jax.vmap(trace_function, (0, 0), 0)(A_i, B_i), 0)

        # Second term
        W_Z = jnp.kron(p.W, tilde_Z)  # W ⊗ Z
        v_X = self.tensor_contraction_with_X(p.v, sp.X)  # v ⋅ X
        v_X = jnp.reshape(jnp.moveaxis(v_X, 2, 1), (K * T, K * T))
        r_X = self.tensor_contraction_with_X(p.r, sp.X)  # r ⋅ X
        r_X = jnp.reshape(jnp.moveaxis(r_X, 2, 1), (K * T, K * T))

        diag_square_xi = jnp.einsum("dk,kl->dkl", sp.bar_xi ** 2, jnp.eye(K))
        Rxi = jnp.einsum("dkl,dlm->dkm", p.R, diag_square_xi )
        RxiRT = jnp.einsum("dkm,dnm->kn", Rxi, p.R)
        RxiRTDelta = jnp.kron(RxiRT, Delta_no_J)
        R_total = jnp.sum(p.R, axis=0)
        Rx = jnp.reshape(jnp.einsum("kl,tl->kt", R_total, sp.bar_x), K * T)
        RxxRT = jnp.outer(Rx, Rx)

        A_2 = jnp.eye(T * K) - 2 * (W_Z + v_X)
        B_2 = jnp.kron(p.L, tilde_Z) + r_X + RxiRTDelta + RxxRT
        second_term = jnp.trace(jnp.linalg.solve(A_2, B_2)) / N

        # Trace terms and contractions
        trace_terms = (
                0.5 * jnp.trace(p.U) +
                0.5 * self.tensor_dot(p.tilde_v, p.r) +
                0.5 * self.tensor_dot(p.p, p.v) +
                jnp.einsum("dkm,dkm->", p.tilde_R, p.R) +
                0.5 * jnp.trace(p.tilde_W @ p.L.T) +
                0.5 * jnp.trace(p.S @ p.W.T)
        )

        return first_term + second_term + trace_terms

    @jit
    def gradient(self, point, ext_field=0):
        """
        Compute the gradient of the free energy with respect to the Point variables.

        Parameters:
        - point: Instance of the Point class, containing optimization variables.

        Returns:
        - grad_point: Point object with the values of the gradient.
        """

        # Extract static parameters
        bar_sigma, bar_e, bar_x = self.bar_sigma, self.bar_e, self.bar_x
        bar_xi, G, Z = self.bar_xi, self.G, self.Z
        Delta, X = self.Delta, self.X

        # Extract point variables
        p = point
        N, T, K, D = bar_sigma.shape[0], Z.shape[0], p.W.shape[0], p.R.shape[0]

        projector = (jnp.eye(T) - jnp.ones((T, T)) / T)
        tilde_Delta = Delta @ projector
        tilde_Z = Z @ projector

        # Add external field for breaking the symmetry
        ext_field_term = jnp.tile( (jnp.eye(K)*ext_field)[jnp.newaxis,:,:], [D,1,1])

        eeT = jnp.einsum("il,im->ilm", bar_e, bar_e)
        xxT = jnp.einsum("tl,um->lmtu", bar_x, bar_x)

        # Trace terms and contractions
        der_W = 0.5 * p.S
        der_S = 0.5 * p.W

        der_U = 0.5 * jnp.eye(K)

        der_R = p.tilde_R
        der_tilde_R = p.R

        der_L = 0.5 * p.tilde_W
        der_tilde_W = 0.5 * p.L

        der_r = 0.5 * p.tilde_v
        der_tilde_v = 0.5 * p.r

        der_p = 0.5 * p.v
        der_v = 0.5 * p.p

        # First term: one based on A_i and B_i
        A_i = jnp.einsum("klmn,ikl->imn", p.tilde_v, eeT)  # tilde_v ⋅ e e^T
        A_i += jnp.einsum("i,kl->ikl", bar_sigma ** 2, p.tilde_W)  # sigma^2 * tilde_W
        A_i += jnp.einsum("i,kl->ikl", jnp.ones(N), p.U)  # U

        invA_i = jax.vmap(inv, 0, 0)(A_i)

        B_i = jnp.einsum("i,kl->ikl", bar_sigma ** 2, p.S)
        B_i += jnp.einsum("klmn,ikl->imn", p.p, eeT)  # p ⋅ e e^T
        B_i -= jnp.einsum("dkl,ilm,dnm,dii->ikn", p.tilde_R, eeT, p.tilde_R, G)

        # "easy" derivatives
        der_S -= jnp.einsum("ikl,i->lk", invA_i, bar_sigma ** 2) / (2 * N)
        der_p -= jnp.einsum("ikl,imn->klnm", eeT, invA_i) / (2 * N)
        der_tilde_R += jnp.einsum("ikl,dlm,imn,dii->dkn", invA_i, p.tilde_R, eeT, G) / (2 * N)
        der_tilde_R += jnp.einsum("ikl,imn,dkn,dii->dlm", invA_i, eeT, p.tilde_R, G) / (2 * N)

        # "complicated" derivatives
        der_tilde_W += jnp.einsum("ikl,imn,ink,i->lm", invA_i, invA_i, B_i, bar_sigma ** 2) / (2 * N)
        der_U += jnp.einsum("ikl,imn,ink->lm", invA_i, invA_i, B_i) / (2 * N)
        der_tilde_v += jnp.einsum("ikl,iop,imn,ink->oplm", invA_i, eeT, invA_i, B_i) / (2 * N)

        # Second term: one based on W_Z and v_X
        W_Z = jnp.kron(p.W, tilde_Z)
        v_X = jnp.reshape(jnp.moveaxis(self.tensor_contraction_with_X(p.v, X), 2, 1), (K * T, K * T))
        r_X = jnp.reshape(jnp.moveaxis(self.tensor_contraction_with_X(p.r, X), 2, 1), (K * T, K * T))
        A_2 = jnp.eye(T * K) - 2 * (W_Z + v_X)
        RxiRT = jnp.einsum("dkm,dm,dnm,ts->knts", p.R, bar_xi ** 2, p.R, tilde_Delta)
        RxxTRT = jnp.einsum("dkl,lmts,cnm->knts", p.R+ext_field_term, xxT, p.R+ext_field_term)

        #InvA_2 = jnp.transpose(jnp.reshape(inv(A_2), (K, T, K, T)), (0, 2, 1, 3))
        InvA_2 = jnp.transpose(jnp.reshape(jnp.linalg.solve(A_2,np.eye(K*T)), (K, T, K, T)), (0, 2, 1, 3))
        B_2 = jnp.transpose(jnp.reshape(jnp.kron(p.L, tilde_Z) + r_X, (K, T, K, T)), (0, 2, 1, 3)) + RxiRT + RxxTRT

        # "easy" derivatives
        der_L += jnp.einsum("kltu,ut->lk", InvA_2, tilde_Z) / N
        der_r += jnp.einsum("kltu,mnut->mnlk", InvA_2, X) / N
        der_R += jnp.einsum("kltu,dln,dn,ut->dkn", InvA_2, p.R, bar_xi ** 2, tilde_Delta) / N
        der_R += jnp.einsum("kltu,dm,ut,dkm->dlm", InvA_2, bar_xi ** 2, tilde_Delta, p.R) / N
        der_R += jnp.einsum("kltu,dlm,mnut,cd->ckn", InvA_2, p.R + ext_field_term, xxT, jnp.ones([D, D])) / N
        der_R += jnp.einsum("kltu,mnut,dkn,cd->clm", InvA_2, xxT, p.R  + ext_field_term, jnp.ones([D, D])) / N

        # "complicated" derivatives
        der_W += jnp.einsum("kltu,ur,mnrs,nkst->lm", InvA_2, tilde_Z, InvA_2, B_2) * (2 / N)
        der_v += jnp.einsum("kltu,opur,mnrs,nkst->oplm", InvA_2, X, InvA_2, B_2) * (2 / N)

        # construct the point from the derivatives
        grad_point = Point(der_W, der_v, der_L, der_r, der_R, der_tilde_W, der_U, der_tilde_v, der_S, der_p,
                           der_tilde_R).as_flat_vector()

        return grad_point

    @jit
    def hessian_AB(self, point, ext_field=0):
        """
        Compute the partial Hessian of the free energy with respect to the Point_old variables, but only the "large" term

        Parameters:
        - point: Instance of the Point_old class, containing optimization variables.

        Returns:
        - dict: Hessian for each variable.
        """
        p = point

        idx_map = p.index_map()

        N = self.bar_sigma.shape[0]
        T = self.Z.shape[0]
        K = p.W.shape[0]
        D = p.R.shape[0]

        diag_square_xi = jnp.einsum("dk,kl->dkl", self.bar_xi**2, jnp.eye(K))

        # Add external field for breaking the symmetry
        ext_field_term = jnp.tile((jnp.eye(K) * ext_field)[jnp.newaxis, :, :], [D, 1, 1])

        # The "large matrix" part.
        Id_min_J = jnp.eye(T) - jnp.ones([T, T]) / T
        tilde_Delta = self.Delta @ Id_min_J
        tilde_Z = self.Z @ Id_min_J
        #tilde_Delta = (tilde_Delta + tilde_Delta.T) / 2

        xi_tilde_Delta = jnp.einsum("dkl,tu->dkltu", diag_square_xi, tilde_Delta)

        # Compute intermediate tensors for v and r contractions
        vX = jnp.tensordot(p.v, self.X, axes=([0, 1], [0, 1])).swapaxes(1, 2).reshape(K * T, K * T)
        rX = jnp.tensordot(p.r, self.X, axes=([0, 1], [0, 1])).swapaxes(1, 2).reshape(K * T, K * T)
        xxT = jnp.einsum("tl,um->lmtu", self.bar_x, self.bar_x)
        RxxT = jnp.einsum("d,kmtu->dkmtu", jnp.ones(D), jnp.einsum("dkl,lmtu->kmtu", p.R + ext_field_term, xxT))  # sum over d as well :)
        RxiDelta = jnp.einsum("dkl,dlmtu->dkmtu", p.R, xi_tilde_Delta)

        # Define and invert matrix A2
        deltaA = - 2 * jnp.kron(p.W, tilde_Z) - 2 * vX
        A = jnp.eye(T * K) + deltaA

        #InvA = jnp.linalg.inv(A)
        InvA = jnp.linalg.solve(A,jnp.eye(K*T))

        InvA_tensor = jnp.swapaxes(jnp.reshape(InvA, (K, T, K, T)), 1, 2)

        # Define matrix B and compute tensor products
        B = (jnp.kron(p.L, self.Z) + rX + jnp.reshape(jnp.einsum("dkmtu,dnm->ktnu", RxxT, p.R + ext_field_term),(K * T, K * T))
             + jnp.reshape(jnp.einsum("dkmtu,dnm->ktnu", RxiDelta, p.R),(K * T, K * T)))
        B_tensor = jnp.reshape(B, (K, T, K, T)).swapaxes(1, 2)

        InvAB_tensor = jnp.swapaxes(jnp.reshape(InvA @ B, (K, T, K, T)), 1, 2)
        InvAZ_tensor = jnp.tensordot(2 * InvA_tensor, tilde_Z, axes=([3], [0]))
        InvAX_tensor = jnp.tensordot(2 * InvA_tensor, self.X, axes=([3], [2]))  # Order of axes: d,d,T,d,d,T
        # Want to switch the order of the axes in InvAX_tensor: put axis 1 after the axis 4, so the new order of axes is [0,2,3,4,1,5]
        InvAX_tensor = jnp.transpose(InvAX_tensor, (0, 2, 3, 4, 1, 5))  # Order of axes: d,T,d,d,d,T
        InvAXInvAB = jnp.tensordot(InvAX_tensor, InvAB_tensor, axes=([5], [2]))  # Order of axes: d,T,d,d,d,d,d,T
        InvAZInvAB = jnp.tensordot(InvAZ_tensor, InvAB_tensor, axes=([3], [2]))  # Order of axes: d,d,T,d,d,T

        # Assembling the Hessian part d^2(InvAB)/dW^2
        #d2W = jnp.tensordot(InvAZ_tensor, InvAZInvAB, axes=([3, 2, 0], [2, 5, 4])) / N  # Add the coefficient of two later because of the square of the term in the Hessian
        d2W = 4*jnp.einsum("abop,pq,cdqr,rs,efst,fato->bcde",InvA_tensor,tilde_Z,InvA_tensor,tilde_Z,InvA_tensor,B_tensor)/N #two times two!
        d2W += jnp.transpose(d2W,(2,3,0,1))
        # Assembling the Hessian part d^2(InvAB)/dv^2
        d2v = jnp.tensordot(InvAX_tensor, InvAXInvAB, axes=(
        [1, 5, 0], [7, 1, 6])) / N  # add the coefficient of two later because of the square of the term in the Hessian.
        # Assembling the Hessian part d^2(InvAB)/dWdv
        dWdv = jnp.tensordot(InvAZ_tensor, InvAXInvAB, axes=([3, 2, 0], [1, 7, 6])) / N + jnp.transpose(
            jnp.tensordot(InvAX_tensor, InvAZInvAB, axes=([0, 1, 5], [4, 5, 2])) / N, (4, 5, 0, 1, 2, 3))

        # Assemble the Hessian part with the derivatives on the W and v
        total_point_size = p.as_flat_vector().shape[0]
        HessianAB = jnp.zeros((total_point_size, total_point_size))

        HessianAB = HessianAB.at[idx_map["W"], idx_map["W"]].set(jnp.reshape(d2W, [p.W.size, p.W.size]))
        HessianAB = HessianAB.at[idx_map["W"], idx_map["v"]].set(jnp.reshape(dWdv, [p.W.size, p.v.size]))
        HessianAB = HessianAB.at[idx_map["v"], idx_map["W"]].set(HessianAB[idx_map["W"], idx_map["v"]].T)
        HessianAB = HessianAB.at[idx_map["v"], idx_map["v"]].set(jnp.reshape(d2v, [p.v.size, p.v.size]))
        HessianAB = HessianAB.at[idx_map["v"], idx_map["v"]].set( HessianAB[idx_map["v"], idx_map["v"]] + HessianAB[idx_map["v"], idx_map["v"]].T)

        # Now, take the part which is square in R: part with xi
        xi_term = jnp.einsum("klts,dmnst->dlmkn",InvA_tensor,xi_tilde_Delta)/N
        #xi_term = jnp.transpose(jnp.tensordot(InvA_tensor, xi_tilde_Delta, axes=([2, 3], [3, 2])) / N, (1, 2, 0, 3))  # We have axes of the size K,K,K,K.
        xi_term = jnp.einsum("cklmn,cd->ckldmn", xi_term, jnp.eye(D))  # Added an identity matrix in D-space.
        xxT_term = jnp.transpose(jnp.tensordot(InvA_tensor, xxT, axes=([2, 3], [3, 2])) / N, (1, 2, 0, 3))
        xxT_term = jnp.einsum("klmn,cd->ckldmn", xxT_term,
                              jnp.ones([D, D]))  # Added an ones matrix in D-space, because we sum over different d.
        d2R = xi_term + xxT_term

        HessianAB = HessianAB.at[idx_map["R"], idx_map["R"]].set(jnp.reshape(d2R, [p.R.size, p.R.size]))
        HessianAB = HessianAB.at[idx_map["R"], idx_map["R"]].set( HessianAB[idx_map["R"], idx_map["R"]] + HessianAB[idx_map["R"], idx_map["R"]].T )

        # Now we have to assemble the cross-terms. First, the cross-term of W and L
        dWdL = jnp.transpose(jnp.tensordot(InvAZ_tensor, InvAZ_tensor / 2, axes=([3, 2], [2, 3])) / N, (1, 2, 3, 0))
        HessianAB = HessianAB.at[idx_map["W"], idx_map["L"]].set(jnp.reshape(dWdL, [p.W.size, p.L.size]))
        HessianAB = HessianAB.at[idx_map["L"], idx_map["W"]].set(HessianAB[idx_map["W"], idx_map["L"]].T)

        # Next, the cross-term of W with r
        dWdr = jnp.transpose(jnp.tensordot(InvAZ_tensor, InvAX_tensor / 2, axes=([3, 2], [1, 5])) / N,
                             (1, 2, 3, 4, 5, 0))
        HessianAB = HessianAB.at[idx_map["W"], idx_map["r"]].set(jnp.reshape(dWdr, [p.W.size, p.r.size]))
        HessianAB = HessianAB.at[idx_map["r"], idx_map["W"]].set(HessianAB[idx_map["W"], idx_map["r"]].T)

        # Next, the cross-term of v with L
        dvdL = jnp.transpose(jnp.tensordot(InvAX_tensor, InvAZ_tensor / 2, axes=([1, 5], [3, 2])) / N,
                             (1, 2, 3, 4, 5, 0))
        HessianAB = HessianAB.at[idx_map["v"], idx_map["L"]].set(jnp.reshape(dvdL, [p.v.size, p.L.size]))
        HessianAB = HessianAB.at[idx_map["L"], idx_map["v"]].set(HessianAB[idx_map["v"], idx_map["L"]].T)


        # Next, the cross-tems of v with r
        dvdr = jnp.transpose(jnp.tensordot(InvAX_tensor, InvAX_tensor / 2, axes=([1, 5], [5, 1])) / N,
                             (1, 2, 3, 4, 5, 6, 7, 0))
        HessianAB = HessianAB.at[idx_map["v"], idx_map["r"]].set(jnp.reshape(dvdr, [p.v.size, p.r.size]))
        HessianAB = HessianAB.at[idx_map["r"], idx_map["v"]].set(HessianAB[idx_map["v"], idx_map["r"]].T)


        # Next, the cross-term of W with R
        # Brute force: einsum
        R_xiDelta = jnp.einsum("dmk,dkl,ts,cd->cdmlts", p.R, diag_square_xi, tilde_Delta, jnp.eye(D))
        RxxT = jnp.einsum("dmk,klts,cd->cdmlts", p.R+ext_field_term, xxT, jnp.ones([D, D]))
        xiDelta_RT = jnp.einsum("dkl,ts,dml,cd->cdkmts", diag_square_xi, tilde_Delta, p.R, jnp.eye(D))
        xxTRT = jnp.einsum("klts,dml,cd->cdkmts", xxT, p.R+ext_field_term, jnp.ones([D, D]))

        dWdR = 2 * jnp.einsum("kltu,ur,mnrs,cdokst->lmdno", InvA_tensor, tilde_Z, InvA_tensor, xxTRT + xiDelta_RT) / N
        dWdR += 2*jnp.einsum("kltu,ur,mnrs,cdnost->lmdko", InvA_tensor, tilde_Z, InvA_tensor, RxxT + R_xiDelta) / N

        HessianAB = HessianAB.at[idx_map["W"], idx_map["R"]].set(jnp.reshape(dWdR, [p.W.size, p.R.size]))
        HessianAB = HessianAB.at[idx_map["R"], idx_map["W"]].set(HessianAB[idx_map["W"], idx_map["R"]].T)

        # Next, the cross-term of v with R
        # Brute force: einsum

        dvdR = 2*jnp.einsum("kltu,mnur,oprs,cdjkst->mnlodpj",InvA_tensor,self.X,InvA_tensor,xxTRT+xiDelta_RT)/N
        dvdR += 2*jnp.einsum("kltu,mnur,oprs,cdpjst->mnlodkj",InvA_tensor,self.X,InvA_tensor,RxxT+R_xiDelta)/N

        HessianAB = HessianAB.at[idx_map["v"], idx_map["R"]].set(jnp.reshape(dvdR, [p.v.size, p.R.size]))
        HessianAB = HessianAB.at[idx_map["R"], idx_map["v"]].set(HessianAB[idx_map["v"], idx_map["R"]].T)
        return HessianAB
    @jit
    def hessian_CD(self, point):
        """
        Compute the partial Hessian of the free energy with respect to the Point_old variables, but only the "i" terms

        Parameters:
        - point: Instance of the Point_old class

        Returns:
        - matrix: Hessian of "i" terms for each variable.
        """

        p = point

        idx_map = p.index_map()

        total_point_size = p.as_flat_vector().shape[0]
        hessian_CD = jnp.zeros((total_point_size, total_point_size))

        # Dimensions
        N, T, K, D = self.bar_sigma.shape[0], self.Z.shape[0], p.W.shape[0], p.R.shape[0]

        # Precompute common terms
        eeT = jnp.einsum("il,im->ilm", self.bar_e, self.bar_e)  # e * e^T

        A_i = jnp.einsum("klmn,ikl->imn",p.tilde_v,eeT)  # v ⋅ e e^T

        A_i += jnp.einsum("i,kl->ikl", self.bar_sigma ** 2, p.tilde_W)  # sigma^2 * tilde_W
        A_i += jnp.einsum("i,kl->ikl", jnp.ones(N), p.U)  # U

        # A_i and B_i
        invA_i = jax.vmap(jnp.linalg.pinv, 0, 0)(A_i) # Batched inverse of A_i

        B_i = jnp.einsum("i,kl->ikl", self.bar_sigma ** 2, p.S)
        B_i += jnp.einsum("klmn,ikl->imn",p.p,eeT)
        tReeT = jnp.einsum("dkl,ilm->idkm", p.tilde_R, eeT)
        tReetR = jnp.einsum("idkm,dnm->idkn", tReeT, p.tilde_R)

        B_i -= jnp.einsum("dii,idkl->ikl", self.G, tReetR)

        invAB_i = jnp.einsum("ikl,ilm->ikm", invA_i, B_i)
        invABinvA_i = jnp.einsum("ikl,ilm->ikm", invAB_i, invA_i)
        large_sum = jnp.einsum("ikl,imn->inklm", invABinvA_i, invA_i)

        large_sum = large_sum + jnp.transpose(large_sum, (0, 3, 4, 1, 2)) # the other order of taking the derivative
        d2U = -1/(2*N)*jnp.sum(large_sum, axis=0)
        d2tildeW = -1/(2*N)*jnp.tensordot(large_sum, self.bar_sigma**4, axes=((0,),(0,)))
        d2tildev = -1/(2*N)*jnp.einsum("iabklmn,icd->abklcdmn", jnp.einsum("iklmn,iab->iabklmn",large_sum,eeT),eeT)

        # rewrite using .at[].set()
        hessian_CD = hessian_CD.at[idx_map["U"], idx_map["U"]].set(jnp.reshape(d2U, [p.U.size, p.U.size]))
        hessian_CD = hessian_CD.at[idx_map["tilde_W"], idx_map["tilde_W"]].set(jnp.reshape(d2tildeW, [p.tilde_W.size, p.tilde_W.size]))
        hessian_CD = hessian_CD.at[idx_map["tilde_v"], idx_map["tilde_v"]].set(jnp.reshape(d2tildev, [p.tilde_v.size, p.tilde_v.size]))

        dtildeWU = -1/(2*N)*jnp.tensordot(large_sum, self.bar_sigma**2, axes=((0,),(0,)))
        hessian_CD = hessian_CD.at[idx_map["tilde_W"], idx_map["U"]].set(jnp.reshape(dtildeWU, [p.tilde_W.size, p.U.size]))
        hessian_CD = hessian_CD.at[idx_map["U"], idx_map["tilde_W"]].set(hessian_CD[idx_map["tilde_W"], idx_map["U"]].T)

        tprod_temp = jnp.einsum("iklmn,iop->iklopmn",large_sum,eeT)
        dtildeWtidldev = -1 / (2 * N) * jnp.tensordot(tprod_temp, self.bar_sigma ** 2, axes=((0,), (0,)))
        hessian_CD = hessian_CD.at[idx_map["tilde_W"], idx_map["tilde_v"]].set(jnp.reshape(dtildeWtidldev, [p.tilde_W.size, p.tilde_v.size]))
        hessian_CD = hessian_CD.at[idx_map["tilde_v"], idx_map["tilde_W"]].set(hessian_CD[idx_map["tilde_W"], idx_map["tilde_v"]].T)


        dtildeUtidldev = -1 / (2 * N) * jnp.tensordot(tprod_temp, jnp.ones(N), axes=((0,), (0,)))
        hessian_CD = hessian_CD.at[idx_map["U"], idx_map["tilde_v"]].set(jnp.reshape(dtildeUtidldev, [p.U.size, p.tilde_v.size]))
        hessian_CD = hessian_CD.at[idx_map["tilde_v"], idx_map["U"]].set(hessian_CD[idx_map["U"], idx_map["tilde_v"]].T)

        # Second derivative w.r.t. tildeR
        large_sum = jnp.einsum("ikl,imn->ilmkn", invA_i, eeT)
        large_sum += jnp.einsum("ikl,imn->iknlm", invA_i, eeT)
        #large_sum = large_sum + jnp.transpose(large_sum, (0, 3, 4, 1, 2))  # the other order of taking the derivative
        d2tildeR = 1/(2*N)*jnp.einsum("dklmn,dc->dklcmn",jnp.einsum("iklmn,dii->dklmn",large_sum,self.G),jnp.eye(D))
        hessian_CD = hessian_CD.at[idx_map["tilde_R"], idx_map["tilde_R"]].set(jnp.reshape(d2tildeR, [p.tilde_R.size, p.tilde_R.size]))
        # Now do the cross-derivatives

        # cross terms of tildeW, U and tildev with S, p, and tildeR
        InvAInvA_i = jnp.einsum("ikl,imn->ilmnk", invA_i, invA_i)

        dtildeWS = 1/(2*N)*jnp.einsum("ilmnk,i->lmnk",InvAInvA_i,self.bar_sigma**4)
        hessian_CD = hessian_CD.at[idx_map["tilde_W"], idx_map["S"]].set(jnp.reshape(dtildeWS, [p.tilde_W.size, p.S.size]))
        hessian_CD = hessian_CD.at[idx_map["S"], idx_map["tilde_W"]].set(hessian_CD[idx_map["tilde_W"], idx_map["S"]].T)

        dtildeUS = 1/(2*N)*jnp.einsum("ilmnk,i->lmnk",InvAInvA_i,self.bar_sigma**2)
        hessian_CD = hessian_CD.at[idx_map["U"], idx_map["S"]].set(jnp.reshape(dtildeUS, [p.U.size, p.S.size]))
        hessian_CD = hessian_CD.at[idx_map["S"], idx_map["U"]].set(hessian_CD[idx_map["U"], idx_map["S"]].T)

        dtildevS = 1/(2*N)*jnp.einsum("ilmnk,iop->oplmnk",InvAInvA_i,jnp.einsum("iop,i->iop",eeT,self.bar_sigma**2))
        hessian_CD = hessian_CD.at[idx_map["tilde_v"], idx_map["S"]].set(jnp.reshape(dtildevS, [p.tilde_v.size, p.S.size]))
        hessian_CD = hessian_CD.at[idx_map["S"], idx_map["tilde_v"]].set(hessian_CD[idx_map["tilde_v"], idx_map["S"]].T)

        dtildeWp =  1/(2*N)*jnp.einsum("ilmnk,iop->lmopnk",InvAInvA_i,jnp.einsum("iop,i->iop",eeT,self.bar_sigma**2))
        hessian_CD = hessian_CD.at[idx_map["tilde_W"], idx_map["p"]].set(jnp.reshape(dtildeWp, [p.tilde_W.size, p.p.size]))
        hessian_CD = hessian_CD.at[idx_map["p"], idx_map["tilde_W"]].set(hessian_CD[idx_map["tilde_W"], idx_map["p"]].T)

        dUp = 1/(2*N)*jnp.einsum("ilmnk,iop->lmopnk",InvAInvA_i,eeT)
        hessian_CD = hessian_CD.at[idx_map["U"], idx_map["p"]].set(jnp.reshape(dUp, [p.U.size, p.p.size]))
        hessian_CD = hessian_CD.at[idx_map["p"], idx_map["U"]].set(hessian_CD[idx_map["U"], idx_map["p"]].T)

        dtildevp = 1/(2*N)*jnp.einsum("ilmnk,ioprs->oplmrsnk",InvAInvA_i,jnp.einsum("iop,irs->ioprs",eeT,eeT))
        hessian_CD = hessian_CD.at[idx_map["tilde_v"], idx_map["p"]].set(jnp.reshape(dtildevp, [p.tilde_v.size, p.p.size]))
        hessian_CD = hessian_CD.at[idx_map["p"], idx_map["tilde_v"]].set(hessian_CD[idx_map["tilde_v"], idx_map["p"]].T)

        dtildeWtildeR = -1/(2*N)*jnp.einsum("i,iab,ice,def,ifg,dii->bcdag", self.bar_sigma**2, invA_i, invA_i, p.tilde_R, eeT, self.G, optimize="optimal")
        dtildeWtildeR += -1/(2*N)*jnp.einsum("i,iab,ice,ifg,dag,dii->bcdef", self.bar_sigma**2, invA_i, invA_i, eeT, p.tilde_R, self.G, optimize="optimal")
        hessian_CD = hessian_CD.at[idx_map["tilde_W"], idx_map["tilde_R"]].set(jnp.reshape(dtildeWtildeR, [p.tilde_W.size, p.tilde_R.size]))
        hessian_CD = hessian_CD.at[idx_map["tilde_R"], idx_map["tilde_W"]].set(hessian_CD[idx_map["tilde_W"], idx_map["tilde_R"]].T)

        dUtildeR = -1/(2*N)*jnp.einsum("iab,ice,def,ifg,dii->bcdag", invA_i, invA_i, p.tilde_R, eeT, self.G, optimize="optimal")
        dUtildeR += -1 / (2 * N) * jnp.einsum("iab,ice,ifg,dag,dii->bcdef", invA_i, invA_i, eeT, p.tilde_R, self.G, optimize="optimal")
        hessian_CD = hessian_CD.at[idx_map["U"], idx_map["tilde_R"]].set(jnp.reshape(dUtildeR, [p.U.size, p.tilde_R.size]))
        hessian_CD = hessian_CD.at[idx_map["tilde_R"], idx_map["U"]].set(hessian_CD[idx_map["U"], idx_map["tilde_R"]].T)

        dtildevtildeR = -1/(2*N)*jnp.einsum("ihj,iab,ice,def,ifg,dii->hjbcdag", eeT, invA_i, invA_i, p.tilde_R, eeT, self.G, optimize="optimal")
        dtildevtildeR += -1/(2*N)*jnp.einsum("ihj,iab,ice,ifg,dag,dii->hjbcdef", eeT, invA_i, invA_i, eeT, p.tilde_R, self.G, optimize="optimal")

        hessian_CD = hessian_CD.at[idx_map["tilde_v"], idx_map["tilde_R"]].set(jnp.reshape(dtildevtildeR, [p.tilde_v.size, p.tilde_R.size]))
        hessian_CD = hessian_CD.at[idx_map["tilde_R"], idx_map["tilde_v"]].set(hessian_CD[idx_map["tilde_v"], idx_map["tilde_R"]].T)


        return hessian_CD

    @jit
    def hessian(self,point, ext_field=0):
        p = point

        idx_map = p.index_map()

        N = self.bar_sigma.shape[0]
        T = self.Z.shape[0]
        K = p.W.shape[0]
        D = p.R.shape[0]

        total_point_size = p.as_flat_vector().shape[0]
        hessian_matrix = jnp.zeros((total_point_size,total_point_size))


        #######################################################
        # First, get the partial hessian from the trace parts #
        #######################################################

        # Since E depends on W and S as 1/2*trace(W@S.T), the cross derivative is 1/2*eye(K) and 1/2*eye(K) respectively

        # rewrite the assignments below using jax.at
        hessian_matrix = hessian_matrix.at[idx_map["W"], idx_map["S"]].set(jnp.eye(K ** 2) / 2)
        hessian_matrix = hessian_matrix.at[idx_map["S"], idx_map["W"]].set(jnp.eye(K ** 2) / 2)

        # Since E depends on U as 1/2*trace(U), there are no cross derivatives
        # Since E depends on tildeR and R as trace(tildeR@R), the cross derivative is eye(K*D**2)
        hessian_matrix = hessian_matrix.at[idx_map["tilde_R"], idx_map["R"]].set(jnp.eye( D*K**2 ))
        hessian_matrix = hessian_matrix.at[idx_map["R"], idx_map["tilde_R"]].set(jnp.eye( D*K**2 ))

        # Since E depends on tildeW and L as 1/2*trace(tildeW@L), the cross derivative is eye(K)/2
        hessian_matrix = hessian_matrix.at[idx_map["tilde_W"], idx_map["L"]].set(jnp.eye(K ** 2) / 2)
        hessian_matrix = hessian_matrix.at[idx_map["L"], idx_map["tilde_W"]].set(jnp.eye(K ** 2) / 2)

        # Since E depends on tildev and r as 1/2*sum(tildev*r), the cross derivative is 1/2*eye(K**2)
        hessian_matrix = hessian_matrix.at[idx_map["tilde_v"], idx_map["r"]].set(jnp.eye(K ** 4) / 2)
        hessian_matrix = hessian_matrix.at[idx_map["r"], idx_map["tilde_v"]].set(jnp.eye(K ** 4) / 2)

        # Since E depends on v and p as 1/2*sum(v*p), the cross derivative is 1/2*eye(K**2)
        hessian_matrix = hessian_matrix.at[idx_map["v"], idx_map["p"]].set(jnp.eye(K ** 4) / 2)
        hessian_matrix = hessian_matrix.at[idx_map["p"], idx_map["v"]].set(jnp.eye(K ** 4) / 2)

        ##################################################
        # Now, get the partial hessian from the sum part #
        ##################################################

        hessian_matrix = hessian_matrix + self.hessian_CD(point)

        ########################################################
        # Finally, get the partial hessian from the large part #
        ########################################################
        hessAB = self.hessian_AB(point, ext_field=ext_field)
        index_rangeAB = hessAB.shape[0]
        hessian_matrix = hessian_matrix.at[:index_rangeAB,:index_rangeAB].set( hessian_matrix[:index_rangeAB,:index_rangeAB] + hessAB)


        return hessian_matrix

    def find_saddle_point(self, init_point, gamma=1, n_steps_max=20, eps=1e-5, ext_field=0):
        """
        Find the saddle point of the free energy with respect to the Point_old variables via Newton's descent.

        Parameters:
        - init_point: Instance of the Point_old class from where we start the optimization.
        - gamma: Step size for the Newton's descent.
        - n_steps_max: Maximum number of steps for the optimization.
        - eps: Tolerance for the convergence.

        Returns:
        - Point_old: The saddle point of the free energy.
        - convergence: Boolean that indicates whether the optimization has converged.
        """

        point = copy.deepcopy(init_point)
        point_new = copy.deepcopy(init_point)

        flattened_point = point.as_flat_vector()
        for i in range(n_steps_max):
            grad_flat = self.gradient(point, ext_field=ext_field)
            hess = self.hessian(point, ext_field=ext_field)
            update_with_pinv = jnp.linalg.pinv(hess, hermitian=True) @ grad_flat
            updated_flattened_point = flattened_point - gamma * update_with_pinv
            point_new.update_from_vector(updated_flattened_point)
            if np.linalg.norm(updated_flattened_point - flattened_point) < eps:
                point = copy.deepcopy(point_new)

                # Flipping signs of v, if accidentally happen to be anti-aligned
                R_diags = np.einsum("dkk->k", point.R)
                c_matrix = np.diag(np.sign(R_diags))
                point.R = c_matrix @ point.R
                point.tilde_R = c_matrix @ point.tilde_R
                point.U = c_matrix @ point.U @ c_matrix
                point.W = c_matrix @ point.W @ c_matrix
                point.tilde_W = c_matrix @ point.tilde_W @ c_matrix
                point.L = c_matrix @ point.L @ c_matrix
                point.S = c_matrix @ point.S @ c_matrix
                point.v = np.einsum("klmn,am,nb->klab", point.v, c_matrix, c_matrix)
                point.tilde_v = np.einsum("klmn,am,nb->klab", point.tilde_v, c_matrix, c_matrix)
                point.p = np.einsum("klmn,am,nb->klab", point.p, c_matrix, c_matrix)
                point.r = np.einsum("klmn,am,nb->klab", point.r, c_matrix, c_matrix)

                return point, True
            point = point_new
            flattened_point = updated_flattened_point

        del point_new
        point = copy.deepcopy(point)

        # Flipping signs of v, if accidentally happen to be anti-aligned
        R_diags = np.einsum("dkk->k", point.R)
        c_matrix = np.diag(np.sign(R_diags))
        point.R = c_matrix @ point.R
        point.tilde_R = c_matrix @ point.tilde_R
        point.U = c_matrix @ point.U @ c_matrix
        point.W = c_matrix @ point.W @ c_matrix
        point.tilde_W = c_matrix @ point.tilde_W @ c_matrix
        point.L = c_matrix @ point.L @ c_matrix
        point.S = c_matrix @ point.S @ c_matrix
        point.v = np.einsum("klmn,am,nb->klab", point.v, c_matrix, c_matrix)
        point.tilde_v = np.einsum("klmn,am,nb->klab", point.tilde_v, c_matrix, c_matrix)
        point.p = np.einsum("klmn,am,nb->klab", point.p, c_matrix, c_matrix)
        point.r = np.einsum("klmn,am,nb->klab", point.r, c_matrix, c_matrix)

        return point, False


    def find_solution(self,eps=10**(-4),verbose=False):
        """
        Find the solution of the free energy by starting from an analytical solution for the noiseless potential,
        then slowly adding noise sources.
        Returns: point: The saddle point of the free energy.
        """

        init_point = self.true_diagonal_solution_no_kernel_difference_no_z_no_xi()
        saved_bar_xi = copy.deepcopy(self.bar_xi)
        saved_bar_sigma = copy.deepcopy(self.bar_sigma)
        saved_X = copy.deepcopy(self.X)
        saved_Xi = copy.deepcopy(self.Xi)

        copied_potential = Potential(saved_bar_sigma*0, self.bar_e, self.G, saved_bar_xi*0, self.Z, self.Delta, self.bar_x, saved_Xi*0)

        # Here we first add the small field to break the symmetry
        ext_field=-10**-3
        init_point, converged = copied_potential.find_saddle_point(init_point, ext_field=ext_field)

        if not converged:
            print("Converged with external field:", converged)

        # Here we slowly add the xi to the potential.
        already_added = 0
        attempted_step_size = 1
        if verbose:
            print("Adding xi...")
            with tqdm(total=100) as pbar:
                while already_added < 1: #and attempted_step_size > 1e-4: # Either we have added all the xi or the step size is too small
                    # Construct the potential with more xi
                    #self.bar_xi = saved_bar_xi * jnp.minimum(1,already_added + attempted_step_size)
                    copied_potential = Potential(saved_bar_sigma * 0, self.bar_e, self.G, saved_bar_xi * jnp.minimum(1,already_added + attempted_step_size), self.Z,
                                                 self.Delta, self.bar_x, saved_Xi * 0)
                    point, converged = copied_potential.find_saddle_point(init_point, ext_field=ext_field)
                    R_diag_elements = jnp.einsum("dii->di", point.R) + np.sqrt(eps)
                    R_diag_elements_old = jnp.einsum("dii->di", init_point.R) + np.sqrt(eps)
                    if not converged:
                        attempted_step_size /= 2
                    #elif jnp.min(R_diag_elements / R_diag_elements_old) < 0.5:  # If the new R has a diagonal element that is less than half of the old one, we stop
                    #    attempted_step_size /= 2
                    #    print("R=", point.R)
                    else:
                        already_added += attempted_step_size
                        init_point = point
                        attempted_step_size *= 2
                        pbar.update(100*already_added - pbar.n)
        else:
            while already_added < 1: #and attempted_step_size > 1e-4: # Either we have added all the xi or the step size is too small
                # Construct the potential with more xi
                copied_potential = Potential(saved_bar_sigma * 0, self.bar_e, self.G,
                                             saved_bar_xi * jnp.minimum(1, already_added + attempted_step_size), self.Z,
                                             self.Delta, self.bar_x, saved_Xi * 0)
                point, converged = copied_potential.find_saddle_point(init_point, ext_field=ext_field)
                R_diag_elements = jnp.einsum("dii->di", point.R) + np.sqrt(eps)
                R_diag_elements_old = jnp.einsum("dii->di", init_point.R) + np.sqrt(eps)
                if not converged:
                    attempted_step_size /= 2
                #elif jnp.min(R_diag_elements / R_diag_elements_old) < 0.5:  # If the new R has a diagonal element that is less than half of the old one, we stop
                #    attempted_step_size /= 2
                else:
                    already_added += attempted_step_size
                    init_point = point
                    attempted_step_size *= 2

        # Here we slowly add the sigma to the potential.
        if verbose:
            print("Adding sigma:")
            eps = 1e-4  # for numerical stability
            already_added = 0.0
            maximal_attempted_step_size = np.min([( np.min(np.std(self.bar_x, axis=0)) / 100 ) / np.max(saved_bar_sigma) + eps, 1])
            print("maximal attempted step size:", maximal_attempted_step_size)
            attempted_step_size = maximal_attempted_step_size
            with tqdm(total=100) as pbar:
                while already_added < 1: # and attempted_step_size > 1e-4:
                    # Construct the potential with more sigma
                    # self.bar_sigma = saved_bar_sigma * jnp.minimum(1, already_added + attempted_step_size)
                    copied_potential = Potential(saved_bar_sigma * jnp.minimum(1, already_added + attempted_step_size),
                                                 self.bar_e, self.G,
                                                 saved_bar_xi,
                                                 self.Z,
                                                 self.Delta, self.bar_x, saved_Xi * 0)
                    point, converged = copied_potential.find_saddle_point(init_point, ext_field=ext_field)
                    R_diag_elements = jnp.einsum("dii->di", point.R) + np.sqrt(eps)
                    R_diag_elements_old = jnp.einsum("dii->di", init_point.R) + np.sqrt(eps)

                    #print("New:", R_diag_elements)
                    #print("Old:", R_diag_elements_old)

                    if not converged:
                        attempted_step_size /= 2
                    elif jnp.min(
                            R_diag_elements / R_diag_elements_old) < 0.5:  # If the new R has a diagonal element that is less than half of the old one, we stop
                        attempted_step_size /= 2
                    else:
                        already_added += attempted_step_size
                        attempted_step_size *= 2
                        attempted_step_size = np.min([attempted_step_size, maximal_attempted_step_size])
                        init_point = point
                        pbar.update(100*already_added - pbar.n)

        else:
            already_added = 0.0
            maximal_attempted_step_size = np.min([(np.min(np.std(self.bar_x, axis=0)) / 100) / np.max(saved_bar_sigma) + eps, 1])
            attempted_step_size = maximal_attempted_step_size
            eps = 1e-4  # for numerical stability
            while already_added < 1:
                # Construct the potential with more sigma
                # self.bar_sigma = saved_bar_sigma * jnp.minimum(1, already_added + attempted_step_size)
                copied_potential = Potential(saved_bar_sigma * jnp.minimum(1, already_added + attempted_step_size),
                                             self.bar_e, self.G,
                                             saved_bar_xi,
                                             self.Z,
                                             self.Delta, self.bar_x, saved_Xi * 0)
                point, converged = copied_potential.find_saddle_point(init_point, ext_field=ext_field)
                R_diag_elements = jnp.einsum("dii->di", point.R) + np.sqrt(eps)
                R_diag_elements_old = jnp.einsum("dii->di", init_point.R) + np.sqrt(eps)

                if not converged:
                    attempted_step_size /= 2
                elif jnp.min(R_diag_elements/R_diag_elements_old) < 0.5: # If the new R has a diagonal element that is less than half of the old one, we stop
                    attempted_step_size /= 2
                else:
                    already_added += attempted_step_size
                    attempted_step_size *= 2
                    attempted_step_size = np.min([attempted_step_size, maximal_attempted_step_size])
                    init_point = point

        # Here we slowly add the X to the potential.
        already_added = 0
        attempted_step_size = 1
        if verbose:
            print("\nAdding X...")
            with tqdm(total=100) as pbar:
                while already_added < 1: # and attempted_step_size > 1e-4:
                    # Construct the potential with more X
                    # self.X = saved_X * jnp.minimum(1, already_added + attempted_step_size)
                    # self.Xi = saved_Xi * jnp.minimum(1, already_added + attempted_step_size)
                    copied_potential = Potential(saved_bar_sigma,
                                                 self.bar_e, self.G,
                                                 saved_bar_xi,
                                                 self.Z,
                                                 self.Delta, self.bar_x,
                                                 saved_Xi * jnp.minimum(1, already_added + attempted_step_size))
                    point, converged = copied_potential.find_saddle_point(init_point, ext_field=ext_field)
                    R_diag_elements = jnp.einsum("dii->di", point.R) + np.sqrt(eps)
                    R_diag_elements_old = jnp.einsum("dii->di", init_point.R) + np.sqrt(eps)
                    if not converged:
                        attempted_step_size /= 2
                    # elif jnp.min(
                    #        R_diag_elements / R_diag_elements_old) < 0.5:  # If the new R has a diagonal element that is less than half of the old one, we stop
                    #    attempted_step_size /= 2
                    else:
                        already_added += attempted_step_size
                        init_point = point
                        attempted_step_size *= 2
                        pbar.update(100*already_added - pbar.n)
        else:
            while already_added < 1:  # and attempted_step_size > 1e-4:
                # Construct the potential with more X
                #self.X = saved_X * jnp.minimum(1, already_added + attempted_step_size)
                #self.Xi = saved_Xi * jnp.minimum(1, already_added + attempted_step_size)
                copied_potential = Potential(saved_bar_sigma,
                                             self.bar_e, self.G,
                                             saved_bar_xi,
                                             self.Z,
                                             self.Delta, self.bar_x,
                                             saved_Xi * jnp.minimum(1, already_added + attempted_step_size))
                point, converged = copied_potential.find_saddle_point(init_point, ext_field=ext_field)
                R_diag_elements = jnp.einsum("dii->di", point.R) + np.sqrt(eps)
                R_diag_elements_old = jnp.einsum("dii->di", init_point.R) + np.sqrt(eps)
                if not converged:
                    attempted_step_size /= 2
                #elif jnp.min(
                #        R_diag_elements / R_diag_elements_old) < 0.5:  # If the new R has a diagonal element that is less than half of the old one, we stop
                #    attempted_step_size /= 2
                else:
                    already_added += attempted_step_size
                    init_point = point
                    attempted_step_size *= 2

        while np.abs(ext_field)>10**-5:
            ext_field = ext_field/2
            init_point, converged = copied_potential.find_saddle_point(init_point, ext_field=ext_field)
            if not converged:
                print("Converged with smaller external field:", converged)

        init_point, converged = copied_potential.find_saddle_point(init_point, ext_field=0)
        print("R without external field:", init_point.R)
        if not converged:
            print("Converged without external field:", converged)
        if verbose:
            print("R=", init_point.R)

        del point

        # Flipping signs of v, if accidentally happen to be anti-aligned
        R_diags = np.einsum("dkk->k", init_point.R)
        c_matrix = np.diag(np.sign(R_diags))
        init_point.R = c_matrix @ init_point.R
        init_point.tilde_R = c_matrix @ init_point.tilde_R
        init_point.U = c_matrix @ init_point.U @ c_matrix
        init_point.W = c_matrix @ init_point.W @ c_matrix
        init_point.tilde_W = c_matrix @ init_point.tilde_W @ c_matrix
        init_point.L = c_matrix @ init_point.L @ c_matrix
        init_point.S = c_matrix @ init_point.S @ c_matrix
        init_point.v = np.einsum("klmn,am,nb->klab", init_point.v, c_matrix, c_matrix)
        init_point.tilde_v = np.einsum("klmn,am,nb->klab", init_point.tilde_v, c_matrix, c_matrix)
        init_point.p = np.einsum("klmn,am,nb->klab", init_point.p, c_matrix, c_matrix)
        init_point.r = np.einsum("klmn,am,nb->klab", init_point.r, c_matrix, c_matrix)

        if np.min(c_matrix.flatten())<0:
            print("Initially was negative. New diagonal:")
            print(np.einsum("dkk->k",init_point.R))

        return init_point

    def true_diagonal_solution_no_kernel_difference_no_z_no_xi(self):
        x_t = self.bar_x
        e_i = self.bar_e
        G = self.G
        Z = self.Z

        T = x_t.shape[0]
        N = e_i.shape[0]
        K = x_t.shape[1]
        D = G.shape[0]
        X = np.zeros([D, D, T, T])
        xi_array = np.zeros(D)
        eeT = np.einsum('ik, il -> ikl', e_i, e_i)
        xxT = np.einsum('tk, ul -> kltu', x_t, x_t)
        diagVarX = np.diag(np.var(x_t, axis=0))

        R = 1 / N * np.einsum('dii, ikl -> dkl', G, eeT)  # normally all different!
        tilde_R = np.einsum("d,kl->dkl", np.ones(D), -2 * T / N * diagVarX)
        U = 2 * T / N * diagVarX

        W = np.zeros([K, K])
        v = np.einsum('ik, lm -> iklm', np.eye(K), np.linalg.pinv(U))
        L = np.zeros([K, K])
        p = np.zeros([K, K, K, K])
        tilde_v = np.zeros([K, K, K, K])
        tilde_W = -2 * (T - 1) / N * np.eye(K)
        r = 1 / N * np.einsum('ikl,imn -> klmn', eeT, eeT)
        S = -(4 * T / N) / T * np.einsum("tu, klut", Z, xxT)

        point = Point(W, v, L, r, R, tilde_W, U, tilde_v, S, p, tilde_R)

        return point

    def generate_sample_data(self, n_samples=10, same_delta_F=False):
        """
        Generate multiple samples of synthetic data.

        Returns:
            data: jnp.ndarray of shape [n_samples, T, N]
        """
        N = self.bar_sigma.shape[0]
        T = self.Z.shape[0]
        K = self.bar_e.shape[1]
        D = self.G.shape[0]

        key = jax.random.PRNGKey(np.random.randint(10 ** 8))
        key, k1 = jax.random.split(key)

        # Shared signal & kernel FFT
        signal_part = jnp.einsum("tk,ik->ti", self.bar_x, self.bar_e)  # Shape [T, N]
        fft_bar_x = jnp.fft.fft(self.bar_x, axis=0)  # Shape [T, K]

        # Generate delta_F (shared or different)
        if jnp.max(jnp.abs(self.Xi)) < T ** (-3):
            delta_Fs = jnp.zeros((n_samples, T, N))
        elif same_delta_F:
            delta_F = jax.random.multivariate_normal(k1, jnp.zeros(T), self.Xi * N, shape=(N,), method='svd').T
            delta_Fs = jnp.tile(delta_F[None, :, :], (n_samples, 1, 1))
        else:
            delta_Fs = jax.random.multivariate_normal(k1, jnp.zeros(T), self.Xi * N, shape=(n_samples, N),
                                                      method='svd')
            delta_Fs = jnp.transpose(delta_Fs, (0, 2, 1))  # [n_samples, T, N]

        # Convolve delta_Fs with bar_x
        fft_delta_F = jnp.fft.fft(delta_Fs, axis=1)  # [n_samples, T, N]
        calcium_part = jnp.fft.ifft(jnp.einsum("sli,tk->slk", fft_delta_F, fft_bar_x), axis=1).real  # [samples, T, K]
        calcium_part = jnp.einsum("slk,ik->sli", calcium_part, self.bar_e)  # [samples, T, N]

        # Trial variability
        key, k2 = jax.random.split(key)
        delta_x = jnp.transpose(
            jax.random.multivariate_normal(k2, jnp.zeros(T), self.Delta, shape=(n_samples, K, D), method='svd'),
            (0, 3, 1, 2)
        )  # [samples, T, K, D]

        delta_x = jnp.einsum("stkd,dk->stkd", delta_x, self.bar_xi)  # scale
        trial_var = jnp.einsum("stkd,dii,ik->sti", delta_x, self.G, self.bar_e)  # [samples, T, N]

        # Noise
        key, k3 = jax.random.split(key)
        noise = jax.random.multivariate_normal(k3, jnp.zeros(T), self.Z, shape=(n_samples, N), method='svd')
        noise = jnp.transpose(noise, (0, 2, 1))  # [samples, T, N]
        noise = jnp.einsum("sti,i->sti", noise, self.bar_sigma * jnp.sqrt(N))

        # Final output
        full_signal = signal_part[None, :, :]  # [1, T, N]
        full_data = full_signal + calcium_part + trial_var + noise  # [samples, T, N]
        return full_data

    def rho(self, point):
        N = self.bar_sigma.shape[0]
        eeT = jnp.einsum("ik,il->ikl", self.bar_e,self.bar_e)
        veeT = jnp.einsum("klmn,ikl->imn",point.tilde_v,eeT)

        matrices_to_invert = jnp.linalg.inv(jnp.einsum("i,kl->ikl",self.bar_sigma**2,point.tilde_W) + point.U[jnp.newaxis,:,:] + veeT)

        term_1 = -1/(2*N) * jnp.einsum("ikl,ilm,imn->ikn", matrices_to_invert,
                                  jnp.einsum("i,kl->ikl",self.bar_sigma**2,point.S)+jnp.einsum("klmn,ikl->imn",point.p,eeT),
                                  matrices_to_invert) + 1/(2*N) * eeT

        intermediate_prod = jnp.einsum("ikl,dii,dlm->ikm", matrices_to_invert,self.G,point.tilde_R)
        term_2 = 1/(2*N) * ( jnp.einsum("ikl,ilm->ikm", intermediate_prod, eeT) + jnp.einsum("ikl,iml->ikm", eeT, intermediate_prod) )
        term_3 = 1/(2*N) * jnp.einsum("ikl,ilm,inm->ikn",intermediate_prod,eeT,intermediate_prod)

        return (term_1+term_2+term_3)*N

    def epsilon(self, point):
        """
            Compute the theoretical expressions for the error epsilon.

                Parameters:
                - point: Instance of the Point_old class.

                Returns:
                - epsilon: The theoretical values for epsilon.
        """

        p = point
        N = self.bar_sigma.shape[0]
        T = self.Z.shape[0]
        K = p.W.shape[0]
        D = p.R.shape[0]
        Id_minus_Ones = (jnp.eye(T) - jnp.ones((T, T)) / T)
        tilde_Delta = self.Delta @ Id_minus_Ones
        tilde_Z = self.Z @ Id_minus_Ones

        xxT = jnp.einsum("tl,um->lmtu", self.bar_x, self.bar_x)

        RxiRT = jnp.einsum("dkm,dm,dnm,ts->knts", p.R, self.bar_xi ** 2, p.R, tilde_Delta)
        RxxTRT = jnp.einsum("dkl,lmts,cnm->knts", p.R, xxT, p.R)
        W_Z = jnp.kron(p.W, tilde_Z)
        v_X = jnp.reshape(jnp.moveaxis(jnp.einsum("klmn,klts->mnts", p.v, self.X), 2, 1), (K * T, K * T))
        A_2 = jnp.eye(T * K) - 2 * (W_Z + v_X)
        InvA_2 = jnp.transpose(jnp.reshape(inv(A_2), (K, T, K, T)), (0, 2, 1, 3))

        B_no_xxT = (jnp.einsum("kl,ts->klts", p.L, tilde_Z) + jnp.einsum("klmn,klts->mnts", p.r, self.X) + RxiRT)
        First_term = -2 * jnp.einsum("klts,dlm,mnst->kn", InvA_2, p.R, xxT)
        Second_term = jnp.einsum("klts,lmsu,mnut->kn", InvA_2 , RxxTRT, InvA_2)
        Third_term = jnp.einsum("klts,lmst->km", B_no_xxT , InvA_2)
        pt1 = (jnp.einsum("kl,ts->klts", p.W, tilde_Z) + jnp.einsum("klmn,klts->mnts", p.v, self.X))
        Fourth_term = 2 * jnp.einsum("klts,lmsu,mnuv,vw,nowt->ko", pt1, InvA_2, B_no_xxT, Id_minus_Ones, InvA_2)
        Fifth_term = jnp.einsum("kltt->kl",xxT)


        epsilon = First_term + Second_term + Third_term + Fourth_term + Fifth_term
        return (epsilon/T)

    def estimate_theory_error_bars_from_signal_variance(self, point, signal_std, delta=0.05):
        """
        Estimate std devs of epsilon and rho due to uncertainty in signal variances,
        via a first-order (delta-method) propagation.

        Inputs
        ------
        point : Point
            Optimized point returned by self.find_solution().
        signal_std : array-like, shape (K,)
            Std deviations of the (per-PC) signal variances you want to propagate.
            (I.e., std of Var[x^{(k)}], one entry per PC.)
        delta : float
            Small multiplicative perturbation (on bar_x columns) used to estimate
            numerical derivatives. Default 0.05.

        Returns
        -------
        epsilon_std   : ndarray, shape (K, K)
            Error bars for epsilon[k, l].
        rho_std       : ndarray, shape (N, K, K)
            Error bars for per-neuron rho[i, k, l].
        mean_rho_std  : ndarray, shape (K, K)
            Error bars for the neuron-mean of rho over axis 0; equals error bars
            for (1 - R).
        """
        import copy
        import numpy as np

        K = self.bar_x.shape[1]
        N = self.bar_e.shape[0]
        signal_std = np.asarray(signal_std, dtype=float)
        assert signal_std.shape == (K,), "signal_std must have shape (K,)"

        # --- Baselines ---
        # epsilon0: (K, K), rho_i0: (N, K, K), mean_rho0: (K, K)
        epsilon0 = np.array(self.epsilon(point))  # (K, K)
        rho_i0 = np.array(self.rho(point))  # (N, K, K)
        sum_rho0 = rho_i0.sum(axis=0)  # (K, K)

        # --- Allocate gradient tensors for delta-method ---
        # ∂ε[a,b] / ∂var_k  -> (K, K, K)
        grad_epsilon = np.zeros((K, K, K), dtype=float)
        # ∂ρ_i[i,a,b] / ∂var_k -> (N, K, K, K)
        grad_rho_i = np.zeros((N, K, K, K), dtype=float)
        # ∂sumρ[a,b] / ∂var_k -> (K, K, K)
        grad_sum_rho = np.zeros((K, K, K), dtype=float)

        # --- Loop over PCs and compute numerical derivatives w.r.t. Var[x^{(k)}] ---
        for k in range(K):
            pert = copy.deepcopy(self)

            # multiplicative bump on the k-th score column: x_k -> (1+delta) x_k
            pert.bar_x = pert.bar_x.at[:, k].set((1.0 + delta) * self.bar_x[:, k])
            pert.update_X()

            point_k = pert.find_solution()

            # re-evaluate observables at the perturbed point
            epsilon_k = np.array(pert.epsilon(point_k))  # (K, K)
            rho_i_k = np.array(pert.rho(point_k))  # (N, K, K)
            sum_rho_k = rho_i_k.sum(axis=0)  # (K, K)

            # actual variance change for that PC (use the empirical var of bar_x[:,k])
            var0 = np.var(np.asarray(self.bar_x[:, k]))
            var1 = np.var(np.asarray(pert.bar_x[:, k]))
            dvar = max(var1 - var0, 1e-20)  # numerical safety

            # finite-difference gradients
            grad_epsilon[:, :, k] = (epsilon_k - epsilon0) / dvar
            grad_rho_i[:, :, :, k] = (rho_i_k - rho_i0) / dvar
            grad_sum_rho[:, :, k] = (sum_rho_k - sum_rho0) / dvar

        # --- Delta method: std ≈ sqrt( sum_k ( ∂f/∂var_k * std[var_k] )^2 ) ---
        # epsilon_std: (K, K)
        epsilon_std = np.sqrt(
            np.sum((grad_epsilon * signal_std[np.newaxis, np.newaxis, :]) ** 2, axis=2)
        )

        # rho_std per neuron: (N, K, K)
        rho_std = np.sqrt(
            np.sum((grad_rho_i * signal_std[np.newaxis, np.newaxis, np.newaxis, :]) ** 2, axis=3)
        )

        # sum over neurons (1 - R): (K, K)
        sum_rho_std = np.sqrt(
            np.sum((grad_sum_rho * signal_std[np.newaxis, np.newaxis, :]) ** 2, axis=2)
        )
        mean_rho_std=sum_rho_std/N

        return epsilon_std, rho_std, mean_rho_std

    def top_eigenvalues(self, point):
        """
        Compute the theoretical expressions for the top eigenvalues of the covariance matrix.

        Parameters:
        - point: Instance of the Point class.

        Returns:
        - eigenvalues: The theoretical values for the top eigenvalues.
        """
        p = point
        N = self.bar_sigma.shape[0]
        T = self.Z.shape[0]
        K = p.W.shape[0]
        D = p.R.shape[0]
        tilde_Delta = self.Delta @ (jnp.eye(T) - jnp.ones((T, T)) / T)
        tilde_Z = self.Z @ (jnp.eye(T) - jnp.ones((T, T)) / T)

        eeT = jnp.einsum("il,im->ilm", self.bar_e, self.bar_e)
        xxT = jnp.einsum("tl,um->lmtu", self.bar_x, self.bar_x)
        diag_bar_xi_squared = jnp.einsum("kl,dk->dkl",jnp.eye(K),self.bar_xi ** 2)
        RxiRT = jnp.einsum("dkl,dlm,dnm,ts->knts", p.R, diag_bar_xi_squared, p.R, tilde_Delta)
        RxxTRT = jnp.einsum("dkl,lmts,cnm->knts", p.R, xxT, p.R)
        W_Z = jnp.kron(p.W, tilde_Z)
        v_X = jnp.reshape(jnp.moveaxis(self.tensor_contraction_with_X(p.v, self.X), 2, 1), (K * T, K * T))
        r_X = jnp.reshape(jnp.moveaxis(self.tensor_contraction_with_X(p.r, self.X), 2, 1), (K * T, K * T))
        A_2 = jnp.eye(T * K) - 2 * (W_Z + v_X)
        InvA_2 = jnp.transpose(jnp.reshape(inv(A_2), (K, T, K, T)), (0, 2, 1, 3))
        B_2 = jnp.transpose(jnp.reshape(jnp.kron(p.L, tilde_Z) + r_X, (K, T, K, T)), (0, 2, 1, 3)) + RxiRT + RxxTRT

        eigenvalues = np.diag(np.einsum("klts,lmsu,mnut->kn",  InvA_2, B_2, InvA_2))/T

        return eigenvalues

    def estimate_signal_variance_uncertainty(self, n_steps = 30, n_samples=100):
        """
        Estimate the distribution of the top K empirical PCA eigenvalues
        by simulating synthetic datasets from the model and applying PCA.

        Returns:
            lambdas: [n_samples, K] array of top PCA eigenvalues
        """
        import copy
        import numpy as np
        from .utils import PCA_matlab_like  # local import to avoid circular dependency

        our_lambdas = self.top_eigenvalues(self.find_solution())
        K = np.shape(self.bar_x)[1]
        N = np.shape(self.bar_e)[0]

        signal_variances_current = np.var(self.bar_x, axis=0)
        synth_data = self.generate_sample_data(n_samples=n_samples)
        # Fit a Gaussian distribution to the lambdas: mean and std
        lambdas = np.zeros((n_samples, K))
        for sample in range(n_samples):
            _, _, eigs = PCA_matlab_like(synth_data[sample, :, :])
            lambdas[sample, :] = eigs[:K] / N # dividing by N due to normalization of vectors as |e|^2=N.

        # Compute the empirical mean and std of the lambdas
        lambdas_mean = np.mean(lambdas, axis=0)
        lambdas_std = np.std(lambdas, axis=0)

        delta_lambda = np.abs(lambdas_mean - our_lambdas) + lambdas_std
        print("Mean lambdas:", lambdas_mean)
        print("Theoretical lambdas:", our_lambdas)


        derivative_matrix = np.zeros((K, K))

        # Now, we will perturb every variance individually by a bit and see how the lambdas change
        for k in range(K):
            potential_copy = copy.deepcopy(self)
            bar_x_updated = np.array(potential_copy.bar_x)
            bar_x_updated[:,k] *= np.sqrt(1.05)
            potential_copy.bar_x = jnp.array(bar_x_updated) # 5% increase
            potential_copy.update_X()
            # Now find the solution
            point = potential_copy.find_solution()
            lambdas_new = potential_copy.top_eigenvalues(point)
            derivative_matrix[:, k] = (lambdas_new - lambdas_mean) / (0.05 * signal_variances_current[k])

        # Now we can estimate the uncertainty in the variances
        std_signal_variance = derivative_matrix @ delta_lambda

        return std_signal_variance


jax.tree_util.register_pytree_node(Point, Point.tree_flatten, Point.tree_unflatten)
jax.tree_util.register_pytree_node(Potential, Potential.tree_flatten, Potential.tree_unflatten)


