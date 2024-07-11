import torch
import torch.nn as nn

from typing import Union

# import pykeops.torch as keops
import torch

import tqdm

class SinkhornSolver(nn.Module):
    """
    Optimal Transport solver under entropic regularisation.
    Based on the code of Gabriel Peyré.
    """
    def __init__(self, epsilon, iterations=100, ground_metric=lambda x: torch.pow(x, 2), reduction='sum'):
        super(SinkhornSolver, self).__init__()
        self.epsilon = epsilon
        self.iterations = iterations
        self.ground_metric = ground_metric
        self.reduction = reduction

    def forward(self, x, y):
        num_x = x.size(-2)
        num_y = y.size(-2)
        
        batch_size = 1 if x.dim() == 2 else x.size(0)

        # Marginal densities are empirical measures
        a = x.new_ones((batch_size, num_x), requires_grad=False) / num_x
        b = y.new_ones((batch_size, num_y), requires_grad=False) / num_y
        
        a = a.squeeze()
        b = b.squeeze()
                
        # Initialise approximation vectors in log domain
        u = torch.zeros_like(a)
        v = torch.zeros_like(b)

        # Stopping criterion
        threshold = 1e-1
        
        # Cost matrix
        C = self._compute_cost(x, y)
        
        # Sinkhorn iterations
        for i in range(self.iterations): 
            u0, v0 = u, v
                        
            # u^{l+1} = a / (K v^l)
            K = self._log_boltzmann_kernel(u, v, C)
            u_ = torch.log(a + 1e-8) - torch.logsumexp(K, dim=1)
            u = self.epsilon * u_ + u
                        
            # v^{l+1} = b / (K^T u^(l+1))
            K_t = self._log_boltzmann_kernel(u, v, C).transpose(-2, -1)
            v_ = torch.log(b + 1e-8) - torch.logsumexp(K_t, dim=1)
            v = self.epsilon * v_ + v
            
            # Size of the change we have performed on u
            diff = torch.sum(torch.abs(u - u0), dim=-1) + torch.sum(torch.abs(v - v0), dim=-1)
            mean_diff = torch.mean(diff)
                        
            if mean_diff.item() < threshold:
                break
   
        #print("Finished computing transport plan in {} iterations".format(i))
    
        # Transport plan pi = diag(a)*K*diag(b)
        K = self._log_boltzmann_kernel(u, v, C)
        pi = torch.exp(K)
        
        # Sinkhorn distance
        if self.reduction == 'sum':
            cost = torch.sum(pi * C, dim=(-2, -1))
        elif self.reduction == 'mean':
            cost = torch.mean(pi * C, dim=(-2, -1))

        return cost, pi

    def _compute_cost(self, x, y):
        x_ = x.unsqueeze(-2)
        y_ = y.unsqueeze(-3)
        C = torch.sum(self.ground_metric(x_ - y_), dim=-1)
        return C

    def _log_boltzmann_kernel(self, u, v, C=None):
        C = self._compute_cost(x, y) if C is None else C
        kernel = -C + u.unsqueeze(-1) + v.unsqueeze(-2)
        kernel /= self.epsilon
        return kernel


def sinkhorn_rpm(x, y, n_iters=10, slack=True, eps=-1):
        ''' Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

        Args:
            log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
            n_iters (int): Number of normalization iterations
            slack (bool): Whether to include slack row and column
            eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

        Returns:
            log(perm_matrix): Doubly stochastic matrix (B, J, K)

        Modified from original source taken from:
            Learning Latent Permutations with Gumbel-Sinkhorn Networks
            https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
        '''

        mmp1 = torch.stack([x] * x.size()[0])
        mmp2 = torch.stack([y] * y.size()[0]).transpose(0, 1)
        log_alpha = torch.sum((mmp1 - mmp2) ** 2, 2).squeeze()

        prev_alpha = None
        if slack:
            zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
            log_alpha_padded = zero_pad(log_alpha[:, None, :, :])
            log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

            for i in range(n_iters):
                # Row normalization
                log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                    dim=1)
                # Column normalization
                log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                    dim=2)
                if eps > 0:
                    if prev_alpha is not None:
                        abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                        if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                            break
                    prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()
            log_alpha = log_alpha_padded[:, :-1, :-1]
        else:
            for i in range(n_iters):
                # Row normalization (i.e. each row sum to 1)
                log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))
                # Column normalization (i.e. each column sum to 1)
                log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))
                if eps > 0:
                    if prev_alpha is not None:
                        abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                        if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                            break
                    prev_alpha = torch.exp(log_alpha).clone()              
        return log_alpha


def sinkhorn(x: torch.Tensor, y: torch.Tensor, p: float = 2,
             w_x: Union[torch.Tensor, None] = None,
             w_y: Union[torch.Tensor, None] = None,
             eps: float = 1e-3,
             max_iters: int = 10, stop_thresh: float = 1e-5,
             verbose=False):
    """
    Compute the Entropy-Regularized p-Wasserstein Distance between two d-dimensional point clouds
    using the Sinkhorn scaling algorithm. This code will use the GPU if you pass in GPU tensors.
    Note that this algorithm can be backpropped through
    (though this may be slow if using many iterations).

    :param x: A [n, d] tensor representing a d-dimensional point cloud with n points (one per row)
    :param y: A [m, d] tensor representing a d-dimensional point cloud with m points (one per row)
    :param p: Which norm to use. Must be an integer greater than 0.
    :param w_x: A [n,] shaped tensor of optional weights for the points x (None for uniform weights). Note that these must sum to the same value as w_y. Default is None.
    :param w_y: A [m,] shaped tensor of optional weights for the points y (None for uniform weights). Note that these must sum to the same value as w_y. Default is None.
    :param eps: The reciprocal of the sinkhorn entropy regularization parameter.
    :param max_iters: The maximum number of Sinkhorn iterations to perform.
    :param stop_thresh: Stop if the maximum change in the parameters is below this amount
    :param verbose: Print iterations
    :return: a triple (d, corrs_x_to_y, corr_y_to_x) where:
      * d is the approximate p-wasserstein distance between point clouds x and y
      * corrs_x_to_y is a [n,]-shaped tensor where corrs_x_to_y[i] is the index of the approximate correspondence in point cloud y of point x[i] (i.e. x[i] and y[corrs_x_to_y[i]] are a corresponding pair)
      * corrs_y_to_x is a [m,]-shaped tensor where corrs_y_to_x[i] is the index of the approximate correspondence in point cloud x of point y[j] (i.e. y[j] and x[corrs_y_to_x[j]] are a corresponding pair)
    """

    if not isinstance(p, int):
        raise TypeError(f"p must be an integer greater than 0, got {p}")
    if p <= 0:
        raise ValueError(f"p must be an integer greater than 0, got {p}")

    if eps <= 0:
        raise ValueError("Entropy regularization term eps must be > 0")

    if not isinstance(p, int):
        raise TypeError(f"max_iters must be an integer > 0, got {max_iters}")
    if max_iters <= 0:
        raise ValueError(f"max_iters must be an integer > 0, got {max_iters}")

    if not isinstance(stop_thresh, float):
        raise TypeError(f"stop_thresh must be a float, got {stop_thresh}")

    if len(x.shape) != 2:
        raise ValueError(f"x must be an [n, d] tensor but got shape {x.shape}")
    if len(y.shape) != 2:
        raise ValueError(f"x must be an [m, d] tensor but got shape {y.shape}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"x and y must match in the last dimension (i.e. x.shape=[n, d], "
                         f"y.shape[m, d]) but got x.shape = {x.shape}, y.shape={y.shape}")

    if w_x is not None:
        if w_y is None:
            raise ValueError("If w_x is not None, w_y must also be not None")
        if len(w_x.shape) > 1:
            w_x = w_x.squeeze()
        if len(w_x.shape) != 1:
            raise ValueError(f"w_x must have shape [n,] or [n, 1] "
                             f"where x.shape = [n, d], but got w_x.shape = {w_x.shape}")
        if w_x.shape[0] != x.shape[0]:
            raise ValueError(f"w_x must match the shape of x in dimension 0 but got "
                             f"x.shape = {x.shape} and w_x.shape = {w_x.shape}")
    if w_y is not None:
        if w_x is None:
            raise ValueError("If w_y is not None, w_x must also be not None")
        if len(w_y.shape) > 1:
            w_y = w_y.squeeze()
        if len(w_y.shape) != 1:
            raise ValueError(f"w_y must have shape [n,] or [n, 1] "
                             f"where x.shape = [n, d], but got w_y.shape = {w_y.shape}")
        if w_x.shape[0] != x.shape[0]:
            raise ValueError(f"w_y must match the shape of y in dimension 0 but got "
                             f"y.shape = {y.shape} and w_y.shape = {w_y.shape}")


    # Distance matrix [n, m]
    x_i = keops.Vi(x)  # [n, 1, d]
    y_j = keops.Vj(y)  # [1, m, d]
    if p == 1:
        M_ij = ((x_i - y_j) ** p).abs().sum(dim=2) # [n, m]
    else:
        M_ij = ((x_i - y_j) ** p).sum(dim=2) ** (1.0 / p)  # [n, m]

    # Weights [n,] and [m,]
    if w_x is None and w_y is None:
        w_x = torch.ones(x.shape[0]).to(x) / x.shape[0]
        w_y = torch.ones(y.shape[0]).to(x) / y.shape[0]
        w_y *= (w_x.shape[0] / w_y.shape[0])

    sum_w_x = w_x.sum().item()
    sum_w_y = w_y.sum().item()
    if abs(sum_w_x - sum_w_y) > 1e-5:
        raise ValueError(f"Weights w_x and w_y do not sum to the same value, "
                         f"got w_x.sum() = {sum_w_x} and w_y.sum() = {sum_w_y} "
                         f"(absolute difference = {abs(sum_w_x - sum_w_y)}")

    log_a = torch.log(w_x)  # [n]
    log_b = torch.log(w_y)  # [m]

    # Initialize the iteration with the change of variable
    u = torch.zeros_like(w_x)
    v = eps * torch.log(w_y)

    u_i = keops.Vi(u.unsqueeze(-1))
    v_j = keops.Vj(v.unsqueeze(-1))

    if verbose:
        pbar = tqdm.trange(max_iters)
    else:
        pbar = range(max_iters)

    for _ in pbar:
        u_prev = u
        v_prev = v

        summand_u = (-M_ij + v_j) / eps
        u = eps * (log_a - summand_u.logsumexp(dim=1).squeeze())
        u_i = keops.Vi(u.unsqueeze(-1))

        summand_v = (-M_ij + u_i) / eps
        v = eps * (log_b - summand_v.logsumexp(dim=0).squeeze())
        v_j = keops.Vj(v.unsqueeze(-1))

        max_err_u = torch.max(torch.abs(u_prev-u))
        max_err_v = torch.max(torch.abs(v_prev-v))
        if verbose:
            pbar.set_postfix({"Current Max Error": max(max_err_u, max_err_v).item()})
        if max_err_u < stop_thresh and max_err_v < stop_thresh:
            break

    P_ij = ((-M_ij + u_i + v_j) / eps).exp()

    approx_corr_1 = P_ij.argmax(dim=1).squeeze(-1)
    approx_corr_2 = P_ij.argmax(dim=0).squeeze(-1)

    if u.shape[0] > v.shape[0]:
        distance = (P_ij * M_ij).sum(dim=1).sum()
    else:
        distance = (P_ij * M_ij).sum(dim=0).sum()
    return distance, approx_corr_1, approx_corr_2


def sinkhorn_cross_batch(x: torch.Tensor, y: torch.Tensor, p: float = 2,
                         w_x: Union[torch.Tensor, None] = None,
                         w_y: Union[torch.Tensor, None] = None,
                         eps: float = 1e-3,
                         max_iters: int = 10, stop_thresh: float = 1e-5,
                         verbose=False):
    """
    Compute the Entropy-Regularized p-Wasserstein Distance between two d-dimensional point clouds
    using the Sinkhorn scaling algorithm. This code will use the GPU if you pass in GPU tensors.
    Note that this algorithm can be backpropped through
    (though this may be slow if using many iterations).

    :param x: A [b, n, d] tensor representing b-batch size sample(s) with d-dimensional point cloud with n points (one per row)
    :param y: A [b, m, d] tensor representing b-batch size sample(s) with d-dimensional point cloud with m points (one per row)
    :param p: Which norm to use. Must be an integer greater than 0.
    :param w_x: A [n,] shaped tensor of optional weights for the points x (None for uniform weights). Note that these must sum to the same value as w_y. Default is None.
    :param w_y: A [m,] shaped tensor of optional weights for the points y (None for uniform weights). Note that these must sum to the same value as w_y. Default is None.
    :param eps: The reciprocal of the sinkhorn entropy regularization parameter.
    :param max_iters: The maximum number of Sinkhorn iterations to perform.
    :param stop_thresh: Stop if the maximum change in the parameters is below this amount
    :param verbose: Print iterations
    :return: a triple (d, corrs_x_to_y, corr_y_to_x) where:
      * d is the approximate p-wasserstein distance between point clouds x and y
      * corrs_x_to_y is a [n,]-shaped tensor where corrs_x_to_y[i] is the index of the approximate correspondence in point cloud y of point x[i] (i.e. x[i] and y[corrs_x_to_y[i]] are a corresponding pair)
      * corrs_y_to_x is a [m,]-shaped tensor where corrs_y_to_x[i] is the index of the approximate correspondence in point cloud x of point y[j] (i.e. y[j] and x[corrs_y_to_x[j]] are a corresponding pair)
    """

    if not isinstance(p, int):
        raise TypeError(f"p must be an integer greater than 0, got {p}")
    if p <= 0:
        raise ValueError(f"p must be an integer greater than 0, got {p}")

    if eps <= 0:
        raise ValueError("Entropy regularization term eps must be > 0")

    if not isinstance(p, int):
        raise TypeError(f"max_iters must be an integer > 0, got {max_iters}")
    if max_iters <= 0:
        raise ValueError(f"max_iters must be an integer > 0, got {max_iters}")

    if not isinstance(stop_thresh, float):
        raise TypeError(f"stop_thresh must be a float, got {stop_thresh}")

    if len(x.shape) != 3:
        raise ValueError(f"x must be an [b, n, d] tensor but got shape {x.shape}")
    if len(y.shape) != 3:
        raise ValueError(f"x must be an [b, m, d] tensor but got shape {y.shape}")
    if x.shape[2] != y.shape[2]:
        raise ValueError(f"x and y must match in the last dimension (i.e. x.shape=[n, d], "
                         f"y.shape[m, d]) but got x.shape = {x.shape}, y.shape={y.shape}")

    if w_x is not None:
        if w_y is None:
            raise ValueError("If w_x is not None, w_y must also be not None")
        if len(w_x.shape) > 1:
            w_x = w_x.squeeze()
        if len(w_x.shape) != 1:
            raise ValueError(f"w_x must have shape [n,] or [n, 1] "
                             f"where x.shape = [n, d], but got w_x.shape = {w_x.shape}")
        if w_x.shape[0] != x.shape[1]:
            raise ValueError(f"w_x must match the shape of x in dimension 0 but got "
                             f"x.shape = {x.shape} and w_x.shape = {w_x.shape}")
    if w_y is not None:
        if w_x is None:
            raise ValueError("If w_y is not None, w_x must also be not None")
        if len(w_y.shape) > 1:
            w_y = w_y.squeeze()
        if len(w_y.shape) != 1:
            raise ValueError(f"w_y must have shape [n,] or [n, 1] "
                             f"where x.shape = [n, d], but got w_y.shape = {w_y.shape}")
        if w_x.shape[0] != x.shape[1]:
            raise ValueError(f"w_y must match the shape of y in dimension 0 but got "
                             f"y.shape = {y.shape} and w_y.shape = {w_y.shape}")


    # Distance matrix [b, n, m]
    x_i = x[:,:,None,:]  # [b, n, 1, d]
    y_j = y[:,None,:,:]  # [b, 1, m, d]
    if p == 1:
        M_ij = ((x_i - y_j) ** p).abs().sum(dim=3) # [b, n, m]
    else:
        M_ij = ((x_i - y_j) ** p).sum(dim=3) ** (1.0 / p)  # [b, n, m]

    # Weights [n,] and [m,]
    if w_x is None and w_y is None:
        w_x = torch.ones(x.shape[0]).to(x.device) / x.shape[0]
        w_y = torch.ones(y.shape[0]).to(x.device) / y.shape[0]
        w_y *= (w_x.shape[0] / w_y.shape[0])

    sum_w_x = w_x.sum().item()
    sum_w_y = w_y.sum().item()
    if abs(sum_w_x - sum_w_y) > 1e-5:
        raise ValueError(f"Weights w_x and w_y do not sum to the same value, "
                         f"got w_x.sum() = {sum_w_x} and w_y.sum() = {sum_w_y} "
                         f"(absolute difference = {abs(sum_w_x - sum_w_y)}")

    log_a = torch.log(w_x)  # [n]
    log_b = torch.log(w_y)  # [m]

    # Initialize the iteration with the change of variable
    u = torch.zeros_like(w_x) # [n]
    v = eps * torch.log(w_y)  # [m]

    u_i = u.unsqueeze(1)[None, :, :] # [n, 1] -> [1, n, 1]
    v_j = v.unsqueeze(0)[None, :, :] # [m, 1] -> [1, 1, m]

    if verbose:
        pbar = tqdm.trange(max_iters)
    else:
        pbar = range(max_iters)

    for _ in pbar:
        u_prev = u
        v_prev = v

        u_i = u_i.to(x.device)
        v_j = v_j.to(x.device)

        summand_u = (-M_ij + v_j) / eps
        summand_u = summand_u.to(x.device)
        u = eps * (log_a - summand_u.logsumexp(dim=2))
        u_i = u[:, :, None]

        summand_v = (-M_ij + u_i) / eps
        v = eps * (log_b - summand_v.logsumexp(dim=1))
        v_j = v[:, None, :]

        max_err_u = torch.max(torch.abs(u_prev-u))
        max_err_v = torch.max(torch.abs(v_prev-v))
        if verbose:
            pbar.set_postfix({"Current Max Error": max(max_err_u, max_err_v).item()})
        if max_err_u < stop_thresh and max_err_v < stop_thresh:
            break

    P_ij = ((-M_ij + u_i + v_j) / eps).exp()
    # [b, n, m] + [b, n, 1] + [b, 1, m]
    # approx_corr_1 = P_ij.argmax(dim=2).squeeze(-1)
    # approx_corr_2 = P_ij.argmax(dim=1).squeeze(-1)

    if u.shape[1] > v.shape[1]:
        distance = (P_ij * M_ij).sum(dim=2).sum()
    else:
        distance = (P_ij * M_ij).sum(dim=1).sum()
    return distance, _, _


if __name__ == '__main__':
    n = torch.randn(2, 588, 256).to("cuda:4")
    m = torch.randn(2, 588, 256).to("cuda:4")
    w_x = torch.randn(1, 588).softmax(dim=-1).to("cuda:4")
    #
    dis, _, _ = sinkhorn_cross_batch(n, m, 2, w_x, w_x)
    print(dis)
