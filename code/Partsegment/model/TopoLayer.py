import numpy as np
import torch
import torch.nn as nn

def point_to_pd(pd):
    """
    :function: calculate the persistence diagram of point cloud

    :param batch_xyz_data:  type array, source points, [B, N, C], where B is batch number of data, N is number of point cloud, C is dim of feature

    :return h1_pd_points: type list, len(B), [...narray(nth_pointcloud_h1pd_points_num, 2)]
            h2_pd_points: type list, len(B), [...narray(nth_pointcloud_h2pd_points_num, 2)]
    """


    # each dim pd of pointcloud in batch
    h1_pd_points = []
    h2_pd_points = []

    for batch_index in range(len(pd)):
        h1_pd_points.append(pd[batch_index][0].to("cuda"))
        h2_pd_points.append(pd[batch_index][1].to("cuda"))


#    return maxmin_normalize_persistence_diagram(h1_pd_points), maxmin_normalize_persistence_diagram(h2_pd_points)
    return h1_pd_points, h2_pd_points
    
def maxmin_normalize_persistence_diagram(pd_points):
   normalized_pd_points = []
    
   for pd_point in pd_points:
       min_vals, _ = torch.min(pd_point, dim=0)
       max_vals, _ = torch.max(pd_point, dim=0)
       range_vals = max_vals - min_vals
        
       # Normalize to the range [0, 1]
       normalized_pd_point = (pd_point - min_vals) / range_vals
       normalized_pd_points.append(normalized_pd_point)

   return normalized_pd_points
    
def norm_normalize_persistence_diagram(pd_points):
   normalized_pd_points = []

   for pd_point in pd_points:

       col_norms = torch.norm(pd_point, dim = 0)
       # Normalize the matrix to the range [0, 1]
       normalized_pd_points.append(pd_point / col_norms)

   return normalized_pd_points

# PD -> vector(PPDTF:RationalHatLayer;ExponentialLayer;RationalLayer;LandscapeLayer;BettiLayer;EntropyLayer;ImageLayer, PDTF:HightOrderGuasssianKernel,GLPKernel)

class PermutationEquivariantLayer(nn.Module):
    """
    :param pd_points          : /*type* list/,
                                /*shape* len(B) [ ... narray(nth_pointcloud_h1pd_points_num, C)]/  where B is batch number of data, N is number of point, C is dim of feature, usually C is 2
           dimension          : a set of permutation equivariant functions of dimension
           perm_op            : a set of permutation operation operations(string, either "max", "min", "sum" or None)
           weight_function    : initializer for the weight matrices of the permutation equivariant operations
           bias_function      : initializer for the biases of the permutation equivariant operations
           gamma_function     : initializer for the Gamma matrices of the permutation equivariant operations

    :return vector: type list, len(B), [...narray(nth_pointcloud_h1pd_points_num, dim)]
    """
    def __init__(self, dimension = [5, 10], perm_op = ["max", None]):
        super(PermutationEquivariantLayer, self).__init__()
        self.dimension = dimension
        self.perm_op = perm_op

        # a layer number of permutation equivariant functions and permutation operation operations
        self.layer_num = len(dimension)
        # PD point feature, usually is 2
        self.before_dim = 2

        # denifition network
        layers = []
        before_dim = self.before_dim
        for dim in self.dimension:
            layers.append(nn.Linear(before_dim, dim))
            before_dim = dim

        self.net = nn.Sequential(
            *layers
        )

        gama_layers = []
        before_dim = self.before_dim
        for index, dim in enumerate(self.dimension):
            if self.perm_op[index] != None:
                gama_layers.append(nn.Linear(before_dim, dim, bias = False))
            before_dim = dim

        self.gama_net = nn.Sequential(
            *gama_layers
        )



    def forward(self, pd_points):
        # batch
        batch = len(pd_points)

        # pd_points -> vector
        vec = []

        # pd_point is list, so we need to deal with each persistence diagram points by "for"
        for batch_index in range(batch):
            # batch_index number pd_point
            inp = pd_points[batch_index]
            # the number of points
            num_pd_point = inp.shape[0]

            A = inp
            # calculate the represent vector of each layers
            gamma_layer_index = 0  # because if perm_op = None, we do not produce a linear without bias to append in list gama_layers. we need to calculate from index 0.
            for layer_index in range(self.layer_num):
                # denifition gamma
                if self.perm_op[layer_index] != None:
                    if self.perm_op[layer_index] == "max":
                        beta = torch.max(A, dim=0, keepdim=True)[0]
                        beta = torch.tile(beta, dims = [num_pd_point, 1])
                    elif self.perm_op[layer_index] == "min":
                        beta = torch.min(A, dim=0, keepdim=True)[0]
                        beta = torch.tile(beta, dims=[num_pd_point, 1])
                    elif self.perm_op[layer_index] == "sum":
                        beta = torch.sum(A, dim=0, keepdim=True)
                        beta = torch.tile(beta, dims=[num_pd_point, 1])
                    else:
                        raise Exception("perm_op should be min, max, or sum")
                    B = self.gama_net[gamma_layer_index](beta)
                    gamma_layer_index = gamma_layer_index + 1
                else:
                    B = 0
                A = self.net[layer_index](A)
                A = A - B
            vec.append(A)
        return vec


class RationalHatLayer(nn.Module):
    """
    :param pd_points          : /*type* list/,
                                /*shape* len(B) [ ... narray(nth_pointcloud_h1pd_points_num, C)]/  where B is batch number of data, N is number of point, C is dim of feature, usually C is 2
           lnum               : Number of rational hat structure elements that will be evaluated on the persistence diagram points.
           lmean_init         : Initializer of the means of the rational hat structure elements
           lr_init            : Initializer of the threshold of the rational hat structure elements
           p                  : Norm parameter

    :return vector: type list, len(B), [...narray(nth_pointcloud_h1pd_points_num, lnum)]
    """
    def __init__(self, p = 2, lnum = 25):
        super(RationalHatLayer, self).__init__()
        self.lnum = lnum
        self.p = p

        # parameter
        self.mu = nn.Parameter(torch.rand(1, 2, self.lnum))
        self.r = nn.Parameter(torch.rand(1, 1))

    def forward(self, pd_points):
        # batch
        batch = len(pd_points)

        # pd_points -> vector
        vec = []

        # pd_point is list, so we need to deal with each persistence diagram points by "for"
        for batch_index in range(batch):
            # batch_index number pd_point
            inp = pd_points[batch_index].unsqueeze(-1)
            norms = torch.norm(inp- self.mu, p = self.p, dim = 1, keepdim = False)
            vec.append(1/(1 + norms) - 1/(1 + torch.abs(torch.abs(self.r) - norms)))
        return vec

class RationalLayer(nn.Module):
    """
    :param pd_points          : /*type* list/,
                                /*shape* len(B) [ ... narray(nth_pointcloud_h1pd_points_num, C)]/  where B is batch number of data, N is number of point, C is dim of feature, usually C is 2
           lnum               : Number of rational structure elements that will be evaluated on the persistence diagram points.
           lmean_init         : Initializer of the means of the rational structure elements.
           lvariance_init     : Initializer of the bandwidths of the rational structure elements.
           lalpha_init        : Initializer of the bandwidths of the rational structure elements.

    :return vector: type list, len(B), [...narray(nth_pointcloud_h1pd_points_num, lnum)]
    """
    def __init__(self, lnum = 25):
        super(RationalLayer, self).__init__()
        self.lnum = lnum

        # parameter
        self.mu = nn.Parameter(torch.rand(1, 2, self.lnum))
        self.sg = nn.Parameter(torch.rand(1, 2, self.lnum))
        self.al = nn.Parameter(torch.rand(1, self.lnum))

    def forward(self, pd_points):
        # batch
        batch = len(pd_points)

        # pd_points -> vector
        vec = []

        # pd_point is list, so we need to deal with each persistence diagram points by "for"
        for batch_index in range(batch):
            # batch_index number pd_point
            inp = pd_points[batch_index].unsqueeze(-1)
            output = torch.multiply(torch.abs(inp - self.mu), torch.abs(self.sg))
            output = torch.sum(output, dim = 1, keepdim = False)
            output = 1 / torch.pow(1 + output, self.al)
            vec.append(output)
        return vec

class ExponentialLayer(nn.Module):
    """
    :param pd_points          : /*type* list/,
                                /*shape* len(B) [ ... narray(nth_pointcloud_h1pd_points_num, C)]/  where B is batch number of data, N is number of point, C is dim of feature, usually C is 2
           lnum               : Number of exponential structure elements that will be evaluated on the persistence diagram points.
           lmean_init         : Initializer of the means of the exponential structure elements.
           lvariance_init     : Initializer of the bandwidths of the exponential structure elements.

    :return vector: type list, len(B), [...narray(nth_pointcloud_h1pd_points_num, lnum)]
    """
    def __init__(self, lnum = 25):
        super(ExponentialLayer, self).__init__()
        self.lnum = lnum

        # parameter
        self.mu = nn.Parameter(torch.rand(1, 2, self.lnum))
        self.sg = nn.Parameter(torch.rand(1, 2, self.lnum))

    def forward(self, pd_points):
        # batch
        batch = len(pd_points)

        # pd_points -> vector
        vec = []

        # pd_point is list, so we need to deal with each persistence diagram points by "for"
        for batch_index in range(batch):
            # batch_index number pd_point
            inp = pd_points[batch_index].unsqueeze(-1) # inp:[num_points, 2, 1]
            output = torch.multiply(torch.square(inp - self.mu), torch.square(self.sg))
            output = torch.sum(-output, dim = 1, keepdim = False)
            output = torch.exp(output)
            vec.append(output)
        return vec

class LandscapeLayer(nn.Module):
    """
    :param pd_points          : /*type* list/,
                                /*shape* len(B) [ ... narray(nth_pointcloud_h1pd_points_num, C)]/  where B is batch number of data, N is number of point, C is dim of feature, usually C is 2
           lsample_num        : Number of exponential structure elements that will be evaluated on the persistence diagram points.
           lsample_init       : Initializer of the samples of the diagonal

    :return vector: type list, len(B), [...narray(nth_pointcloud_h1pd_points_num, lsample_num)]
    """
    def __init__(self, lsample_num = 100):
        super(LandscapeLayer, self).__init__()
        self.lsample_num = lsample_num

        # parameter
        self.sp = nn.Parameter(torch.rand(1, self.lsample_num))

    def forward(self, pd_points):
        # batch
        batch = len(pd_points)
        
        device = pd_points[0].device

        # pd_points -> vector
        vec = []

        # pd_point is list, so we need to deal with each persistence diagram points by "for"
        for batch_index in range(batch):
            # batch_index number pd_point
            inp = pd_points[batch_index]
            output = .5 * (inp[:, 1:2] - inp[:, 0:1]) - torch.abs(self.sp - .5 * (inp[:, 1:2] + inp[:, 0:1]))
            output = torch.maximum(output, torch.tensor([0], device=device))
            vec.append(output)
        return vec

class BettiLayer(nn.Module):
    """
    :param pd_points          : /*type* list/,
                                /*shape* len(B) [ ... narray(nth_pointcloud_h1pd_points_num, C)]/  where B is batch number of data, N is number of point, C is dim of feature, usually C is 2
           lsample_num        : Number of samples of the diagonal that will be evaluated on the Betti curves.
           lsample_init       : Initializer of the samples of the diagonal.
           theta              : Sigmoid parameter used for approximating the piecewise constant functions associated to the persistence diagram points.

    :return vector: type list, len(B), [...narray(nth_pointcloud_h1pd_points_num, lsample_num)]
    """
    def __init__(self, lsample_num = 100, theta = 10):
        super(BettiLayer, self).__init__()
        self.lsample_num = lsample_num
        self.theta = theta

        # parameter
        self.sp = nn.Parameter(torch.rand(1, self.lsample_num))

    def forward(self, pd_points):
        # batch
        batch = len(pd_points)

        # pd_points -> vector
        vec = []

        # pd_point is list, so we need to deal with each persistence diagram points by "for"
        for batch_index in range(batch):
            # batch_index number pd_point
            inp = pd_points[batch_index]
            Y, X = inp[:, 1:2], inp[:, 0:1]
            output = .5 * (Y - X) - torch.abs(self.sp - .5 * (Y + X))
            output = 1. / (1. + torch.exp(-self.theta * output))
            vec.append(output)
        return vec

class EntropyLayer(nn.Module):
    """
    :param pd_points          : /*type* list/,
                                /*shape* len(B) [ ... narray(nth_pointcloud_h1pd_points_num, C)]/  where B is batch number of data, N is number of point, C is dim of feature, usually C is 2
           lsample_num        : Number of samples on the diagonal that will be evaluated on the persistence entropies
           lsample_init       : Initializer of the samples of the diagonal.
           theta              : Sigmoid parameter used for approximating the piecewise constant functions associated to the persistence diagram points.

    :return vector: type list, len(B), [...narray(nth_pointcloud_h1pd_points_num, lsample_num)]
    """
    def __init__(self, lsample_num = 100, theta = 10):
        super(EntropyLayer, self).__init__()
        self.lsample_num = lsample_num
        self.theta = theta

        # parameter
        self.sp = nn.Parameter(torch.rand(1, self.lsample_num))

    def forward(self, pd_points):
        # batch
        batch = len(pd_points)
        
        device = pd_points[0].device

        # pd_points -> vector
        vec = []

        # pd_point is list, so we need to deal with each persistence diagram points by "for"
        for batch_index in range(batch):
            # batch_index number pd_point
            inp = pd_points[batch_index]
            Y, X, L = inp[:, 1:2], inp[:, 0:1], inp[:, 1:2] - inp[:, 0:1] + 1e-4
            LN = torch.multiply(L, 1. / torch.matmul(L[:, 0], torch.ones(L.shape[0], device=device)).unsqueeze(-1))
            entropy_terms = torch.where(LN > 0., -torch.multiply(LN, torch.log(LN)), LN)
            output = torch.multiply(entropy_terms, 1. / (1. + torch.exp( -self.theta * (.5 * (Y - X) - torch.abs(self.sp - .5 * (Y + X) ) ) )))
            vec.append(output)
        return vec

class ImageLayer(nn.Module):
    """
    :param pd_points          : /*type* list/,
                                /*shape* len(B) [ ... narray(nth_pointcloud_h1pd_points_num, C)]/  where B is batch number of data, N is number of point, C is dim of feature, usually C is 2
           image_size         : Persistence image size
           image_bnds         : Persistence image boundaries. It is a tuple containing two tuples, each containing the minimum and maximum values of each axis of the plane.
           lvariance_init     : Initializer for the bandwidths of the Gaussian functions centered on the persistence image pixels.

    :return vector: type list, len(B), [...narray(nth_pointcloud_h1pd_points_num, image_size, image_size)]
    """
    def __init__(self, image_size = (20, 20), image_bnds = ((-.001, 1.001), (-.001, 1.001))):
        super(ImageLayer, self).__init__()
        self.image_size = image_size
        self.image_bnds = image_bnds

        self.sg = nn.Parameter(torch.rand(1))

    def forward(self, pd_points):
        # batch
        batch = len(pd_points)
        
        device = pd_points[0].device

        # pd_points -> vector
        vec = []

        # pd_point is list, so we need to deal with each persistence diagram points by "for"
        for batch_index in range(batch):
            # batch_index number pd_point
            inp = pd_points[batch_index]
            bp_inp = torch.matmul(inp, torch.tensor([[1., -1.], [0., 1.]], device=device))
            Y, X, L = inp[:, 1:2], inp[:, 0:1], inp[:, 1:2] - inp[:, 0:1]
            num_points, dim = inp.shape[0], inp.shape[1]
            # 网格坐标
            coords = [torch.arange(self.image_bnds[i][0], self.image_bnds[i][1], (self.image_bnds[i][1] - self.image_bnds[i][0]) / self.image_size[i], device=device) for i in range(dim)]
            M = torch.meshgrid(*coords, indexing='ij')
            mu = torch.concat([tensor.unsqueeze(0) for tensor in M], dim = 0)
            bc_inp = bp_inp.view(num_points, dim, *([1] * dim))
            # calculate the gauss function value of ith pd point in each mesh point
            exponent = -torch.square(bc_inp - mu) / (2 * torch.square(self.sg))
            output = torch.exp(torch.sum(exponent, dim = 1, keepdim = False)) / (2 * np.pi * torch.square(self.sg))
            output = output.view(-1, self.image_size[0] * self.image_size[1])
            vec.append(output)
        return vec


class HightOrderGuasssianKernel(nn.Module):
    """
    :param pd_points            : /*type* list/,
                                  /*shape* len(B) [ ... narray(nth_pointcloud_h1pd_points_num, C)]/  where B is batch number of data, N is number of point, C is dim of feature, usually C is 2
           dim_feature          : the dim of represent vector
           p1                   : pow parameter
           p2                   : pow parameter
           kernel_mean_init     : Initializer for the mean of the Gaussian functions
           kernel_variance_init : Initializer for the variance of the Gaussian functions

    :return vector: type list, len(B), [...narray(nth_pointcloud_h1pd_points_num, dim_feature)]
    """
    def __init__(self, dim_feature = 10, p1 = 2, p2 = 2):
        super(HightOrderGuasssianKernel, self).__init__()
        self.dim_feature = dim_feature
        self.p1 = p1
        self.p2 = p2
        self.kernel_mean = nn.Parameter(torch.rand(self.dim_feature, 1, 2))
        self.kernel_variance = nn.Parameter(torch.rand(self.dim_feature, 1, 2))

    def forward(self, pd_points):
        # batch
        batch = len(pd_points)
        
        device = pd_points[0].device

        # pd_points -> vector
        vec = []

        # pd_point is list, so we need to deal with each persistence diagram points by "for"
        for batch_index in range(batch):
            # batch_index number pd_point
            inp = pd_points[batch_index]
            inp = inp.unsqueeze(0)
            bp_inp = torch.matmul(inp, torch.tensor([[1., -1.], [0., 1.]], device=device))
            exponent = torch.square(bp_inp - self.kernel_mean)
            exponent = -torch.pow(exponent[:, :, 0] / torch.square(self.kernel_variance[:, :, 0]), self.p1) - torch.pow(exponent[:, :, 1] / torch.square(self.kernel_variance[:, :, 1]), self.p2)
            output = torch.exp(exponent)
            output = output.T
            vec.append(output)
        return vec

class GLPKernel(nn.Module):
    """
    :param pd_points            : /*type* list/,
                                  /*shape* len(B) [ ... narray(nth_pointcloud_h1pd_points_num, C)]/  where B is batch number of data, N is number of point, C is dim of feature, usually C is 2
           dim_feature          : the dim of represent vector
           kernel_mean_init     : Initializer for the mean of the Gaussian functions
           kernel_variance_init : Initializer for the variance of the Gaussian functions

    :return vector: type list, len(B), [...narray(nth_pointcloud_h1pd_points_num, dim_feature)]
    """
    def __init__(self, dim_feature = 10):
        super(GLPKernel,self).__init__()
        self.dim_feature = dim_feature
        self.kernel_variance1 = nn.Parameter(torch.rand(self.dim_feature, 1, 2))
        self.kernel_mean1 = nn.Parameter(torch.rand(self.dim_feature, 1, 2))
        self.kernel_variance2 = nn.Parameter(torch.rand(self.dim_feature, 1,  2))
        self.kernel_mean2 = nn.Parameter(torch.rand(self.dim_feature, 1, 2))
        self.variance = nn.Parameter(torch.rand(self.dim_feature, 1))

    def forward(self, pd_points):
        # batch
        batch = len(pd_points)
        
        device = pd_points[0].device

        # pd_points -> vector
        vec = []

        # pd_point is list, so we need to deal with each persistence diagram points by "for"
        for batch_index in range(batch):
            # batch_index number pd_point
            inp = pd_points[batch_index]
            inp = inp.unsqueeze(0)
            bp_inp = torch.matmul(inp, torch.tensor([[1., -1.], [0., 1.]], device=device))

            exponent1 = torch.square(bp_inp[:, :, 0:1] - self.kernel_mean1)
            exponent2 = torch.square(bp_inp[:, :, 1:2] - self.kernel_mean2)

            exponent1 = -2 * torch.pow(torch.sin(torch.pi * exponent1[:, :, 0] / self.kernel_variance1[:, :, 0]), 2) - (
                        exponent1[:, :, 1] / 2 * torch.pow(self.kernel_variance1[:, :, 1], 2))
            exponent2 = -2 * torch.pow(torch.sin(torch.pi * exponent2[:, :, 0] / self.kernel_variance2[:, :, 0]), 2) - (
                        exponent2[:, :, 1] / 2 * torch.pow(self.kernel_variance2[:, :, 1], 2))
            output = torch.pow(self.variance, 2) * torch.multiply(torch.exp(exponent1), torch.exp(exponent2))
            output = output.T
            vec.append(output)
        return vec


class WeightLayer(nn.Module):
    """
    :param pd_points            : /*type* list/,
                                  /*shape* len(B) [ ... narray(nth_pointcloud_h1pd_points_num, C)]/  where B is batch number of data, N is number of point, C is dim of feature, usually C is 2
           vec                :   /*type* list/ the pd representation vector
                                  /*shape* len(B) [ ... narray(nth_pointcloud_h1pd_points_num, C)]/  where B is batch number of data, N is number of point, C is dim of feature
           pweigh               : the kind of weight, including power, grid, gmix
           pweight_init         : Initializer for parameter
           pweight_variance_init: Initializer for the variance of the Gaussian functions
           w_power              : Integer used for exponentiating the distances to the diagonal of the persistence diagram points.
           w_grid_shape         : Grid size of the grid weight function.
           w_grid_bnds          : Grid boundaries of the grid weight function.
           w_guass_num          : Number of Gaussian functions of the mixture of Gaussians weight function.

    :return w_vec: type list, len(B), [...narray(nth_pointcloud_h1pd_points_num, dim_feature)]
    """
    def __init__(self, pweight = "power", w_power = -1, w_grid_shape = -1, w_grid_bnds = -1, w_guass_num = -1):
        super(WeightLayer, self).__init__()
        self.pweight = pweight
        if self.pweight == "power":
            if w_power == -1:
                raise Exception("please input the power (int)")
            self.weight = nn.Parameter(torch.rand(1))
            #self.weight = 1
            self.w_power = w_power
        elif self.pweight == "grid":
            if w_grid_shape == -1:
                raise Exception("please input the size of grid (list), such as [2, 2] (list)")
            elif w_grid_bnds == -1:
                raise Exception("please input the boundary of grid ,such as [[-2.01, 2.01],[-2.01, 2.01]](list)")
            self.weight = nn.Parameter(torch.ones(w_grid_shape))
            self.w_grid_shape = w_grid_shape
            self.w_grid_bnd = w_grid_bnds
        elif self.pweight == "gmix":
            if w_guass_num == -1:
                raise Exception("please input the number of guass functions (int)")
            self.mean_weight = nn.Parameter(torch.rand(2, w_guass_num))
            self.variance_weight = nn.Parameter(torch.rand(2, w_guass_num))
            self.w_guass_num = w_guass_num
        else:
            raise Exception("pweigh should be power, grid or gmix")

    def forward(self, pd_vectors, pd_points):
        # batch
        batch = len(pd_vectors)

        # pd_vectors -> weight_vectors
        w_vec = []

        # pd_vectors is list, so we need to deal with each persistence diagram points by "for"
        for batch_index in range(batch):
            inp = pd_vectors[batch_index]
            inp_pd = pd_points[batch_index]
            num_points, dim_feature = inp.shape[0], inp.shape[1]

            # denifition weight
            if self.pweight == "power":
                weight = self.weight * torch.pow(inp_pd[:, 1:2] - inp_pd[:, 0:1], self.w_power)

            elif self.pweight == "grid":
                grid_shape = self.w_grid_shape
                indices = []

                # get the coordinate of grid for each pd_point
                for dim in range(2):
                    [m, M] = self.w_grid_bnd[dim]
                    coords = inp_pd[:, dim].unsqueeze(-1)
                    # be sure about coords in the range of [m, M]
                    ids = torch.round(grid_shape[dim] * (coords - m) / (M - m))
                    ids = torch.clamp(ids, min=0, max=grid_shape[dim] - 1)
                    indices.append(ids.to(torch.int64))
                indices = torch.cat(indices, dim=1)
                weight = self.weight[indices[:,0], indices[:, 1]].unsqueeze(-1)

            elif self.pweight == "gmix":
                mean_g = self.mean_weight.unsqueeze(0)
                variance_g = self.variance_weight.unsqueeze(0)

                inp_pd = inp_pd.unsqueeze(-1)
                exponent = torch.sum(-torch.multiply(torch.square(inp_pd - mean_g), torch.square(variance_g)), dim = 1)
                weight = torch.sum(torch.exp(exponent), dim = 1).unsqueeze(-1)
            w_vec.append(inp * weight)
        return w_vec

class TopoLayer(nn.Module):
    # PD data
    # PD -> vector(PPDTF:RationalHatLayer;ExponentialLayer;RationalLayer;LandscapeLayer;BettiLayer;EntropyLayer;ImageLayer, PDTF:HightOrderGuasssianKernel,GLPKernel)
    # vector -> w * vector
    # w_vec -> opt(w_vec)(remark topology_feature)
    def __init__(self, PermutationEquivariantLayer_parameters, RationalHatLayer_parameters, ExponentialLayer_parameters, \
        RationalLayer_parameters, LandscapeLayer_parameters, BettiLayer_parameters, EntropyLayer_parameters, \
        ImageLayer_parameters, HightOrderGuasssianKernel_parameters, GLPKernel_parameters, *args):
        super(TopoLayer, self).__init__()
        # init parameter
        self.PermutationEquivariantLayer_parameters = PermutationEquivariantLayer_parameters
        self.RationalHatLayer_parameters = RationalHatLayer_parameters
        self.ExponentialLayer_parameters = ExponentialLayer_parameters
        self.RationalLayer_parameters = RationalLayer_parameters
        self.LandscapeLayer_parameters = LandscapeLayer_parameters
        self.BettiLayer_parameters = BettiLayer_parameters
        self.EntropyLayer_parameters = EntropyLayer_parameters
        self.ImageLayer_parameters = ImageLayer_parameters
        self.HightOrderGuasssianKernel_parameters = HightOrderGuasssianKernel_parameters
        self.GLPKernel_parameters = GLPKernel_parameters

        self.permutation_equivariant_layers = nn.ModuleList()
        self.exponential_layers = nn.ModuleList()
        self.rational_layers = nn.ModuleList()
        self.rational_hat_layers = nn.ModuleList()
        self.landscape_layers = nn.ModuleList()
        self.betti_layers = nn.ModuleList()
        self.entropy_layers = nn.ModuleList()
        self.image_layers = nn.ModuleList()
        self.hight_order_guassian_kernel_layers = nn.ModuleList()
        self.glp_kernel_layers = nn.ModuleList()


        if PermutationEquivariantLayer_parameters != None:
            self.permutation_equivariant_weight_layers = nn.ModuleList()
            if PermutationEquivariantLayer_parameters["permutation_equivariant_layers_nums"] != len(PermutationEquivariantLayer_parameters["weight_layers_parameters"]):
                raise Exception("Please make sure the number of PermutationEquivariantLayers should be equal to the number of WeightLayers")
            for i in range(PermutationEquivariantLayer_parameters["permutation_equivariant_layers_nums"]):
                self.permutation_equivariant_layers.append(PermutationEquivariantLayer(*PermutationEquivariantLayer_parameters["permutation_equivariant_layers_parameters"][i].values()))
                self.permutation_equivariant_weight_layers.append(WeightLayer(*PermutationEquivariantLayer_parameters["weight_layers_parameters"][i].values()))

        if RationalHatLayer_parameters != None:
            self.rational_hat_weight_layers = nn.ModuleList()
            if RationalHatLayer_parameters["rational_hat_layers_nums"] != len(RationalHatLayer_parameters["weight_layers_parameters"]):
                raise Exception("Please make sure the number of RationalHatLayers should be equal to the number of WeightLayers")
            for i in range(RationalHatLayer_parameters["rational_hat_layers_nums"]):
                self.rational_hat_layers.append(RationalHatLayer(*RationalHatLayer_parameters["rational_hat_layers_parameters"][i].values()))
                self.rational_hat_weight_layers.append(WeightLayer(*RationalHatLayer_parameters["weight_layers_parameters"][i].values()))

        if ExponentialLayer_parameters != None:
            self.exponential_weight_layers = nn.ModuleList()
            if ExponentialLayer_parameters["exponential_layers_nums"] != len(ExponentialLayer_parameters["weight_layers_parameters"]):
                raise Exception("Please make sure the number of ExponentialLayers should be equal to the number of WeightLayers")
            for i in range(ExponentialLayer_parameters["exponential_layers_nums"]):
                self.exponential_layers.append(ExponentialLayer(*ExponentialLayer_parameters["exponential_layers_parameters"][i].values()))
                self.exponential_weight_layers.append(WeightLayer(*ExponentialLayer_parameters["weight_layers_parameters"][i].values()))

        if RationalLayer_parameters != None:
            self.rational_weight_layers = nn.ModuleList()
            if RationalLayer_parameters["rational_layers_nums"] != len(RationalLayer_parameters["weight_layers_parameters"]):
                raise Exception("Please make sure the number of RationalLayers should be equal to the number of WeightLayers")
            for i in range(RationalLayer_parameters["rational_layers_nums"]):
                self.rational_layers.append(RationalLayer(*RationalLayer_parameters["rational_layers_parameters"][i].values()))
                self.rational_weight_layers.append(WeightLayer(*RationalLayer_parameters["weight_layers_parameters"][i].values()))

        if LandscapeLayer_parameters != None:
            self.landscape_weight_layers = nn.ModuleList()
            if LandscapeLayer_parameters["landscape_layers_nums"] != len(LandscapeLayer_parameters["weight_layers_parameters"]):
                raise Exception("Please make sure the number of LandscapeLayers should be equal to the number of WeightLayers")
            for i in range(LandscapeLayer_parameters["landscape_layers_nums"]):
                self.landscape_layers.append(LandscapeLayer(*LandscapeLayer_parameters["landscape_layers_parameters"][i].values()))
                self.landscape_weight_layers.append(WeightLayer(*LandscapeLayer_parameters["weight_layers_parameters"][i].values()))

        if BettiLayer_parameters != None:
            self.betti_weight_layers = nn.ModuleList()
            if BettiLayer_parameters["betti_layers_nums"] != len(BettiLayer_parameters["weight_layers_parameters"]):
                raise Exception("Please make sure the number of BettiLayers should be equal to the number of WeightLayers")
            for i in range(BettiLayer_parameters["betti_layers_nums"]):
                self.betti_layers.append(BettiLayer(*BettiLayer_parameters["betti_layers_parameters"][i].values()))
                self.betti_weight_layers.append(WeightLayer(*BettiLayer_parameters["weight_layers_parameters"][i].values()))

        if EntropyLayer_parameters != None:
            self.entropy_weight_layers = nn.ModuleList()
            if EntropyLayer_parameters["entropy_layers_nums"] != len(EntropyLayer_parameters["weight_layers_parameters"]):
                raise Exception("Please make sure the number of EntropyLayers should be equal to the number of WeightLayers")
            for i in range(EntropyLayer_parameters["entropy_layers_nums"]):
                self.entropy_layers.append(EntropyLayer(*EntropyLayer_parameters["entropy_layers_parameters"][i].values()))
                self.entropy_weight_layers.append(WeightLayer(*EntropyLayer_parameters["weight_layers_parameters"][i].values()))

        if ImageLayer_parameters != None:
            self.image_weight_layers = nn.ModuleList()
            if ImageLayer_parameters["image_layers_nums"] != len(ImageLayer_parameters["weight_layers_parameters"]):
                raise Exception("Please make sure the number of ImageLayers should be equal to the number of WeightLayers")
            for i in range(ImageLayer_parameters["image_layers_nums"]):
                self.image_layers.append(ImageLayer(*ImageLayer_parameters["image_layers_parameters"][i].values()))
                self.image_weight_layers.append(WeightLayer(*ImageLayer_parameters["weight_layers_parameters"][i].values()))

        if HightOrderGuasssianKernel_parameters != None:
            self.hight_order_guassian_kernel_weight_layers = nn.ModuleList()
            if HightOrderGuasssianKernel_parameters["hight_order_guassian_kernel_layers_nums"] != len(HightOrderGuasssianKernel_parameters["weight_layers_parameters"]):
                raise Exception("Please make sure the number of HightOrderGuasssianKernel should be equal to the number of WeightLayers")
            for i in range(HightOrderGuasssianKernel_parameters["hight_order_guassian_kernel_layers_nums"]):
                self.hight_order_guassian_kernel_layers.append(HightOrderGuasssianKernel(*HightOrderGuasssianKernel_parameters["hight_order_guassian_kernel_layers_parameters"][i].values()))
                self.hight_order_guassian_kernel_weight_layers.append(WeightLayer(*HightOrderGuasssianKernel_parameters["weight_layers_parameters"][i].values()))

        if GLPKernel_parameters != None:
            self.glp_kernel_weight_layers = nn.ModuleList()
            if GLPKernel_parameters["glp_kernel_layers_nums"] != len(GLPKernel_parameters["weight_layers_parameters"]):
                raise Exception("Please make sure the number of GLPKernel should be equal to the number of WeightLayers")
            for i in range(GLPKernel_parameters["glp_kernel_layers_nums"]):
                self.glp_kernel_layers.append(GLPKernel(*GLPKernel_parameters["glp_kernel_layers_parameters"][i].values()))
                self.glp_kernel_weight_layers.append(WeightLayer(*GLPKernel_parameters["weight_layers_parameters"][i].values()))

    def operation_vector(self, w_vec, perm_op="topk", k=-1):
        """
        :param w_vec: /*type* list/ the resulting weighted pd representation vector
                      /*shape* len(B) [ ... narray(nth_pointcloud_h1pd_points_num, C)]/  where B is batch number of data, N is number of point cloud, C is dim of feature

        :return: topo_feature
        """
        # batch
        batch = len(w_vec)

        # feature dim
        dim = w_vec[0].shape[1]

        # pd_vectors -> weight_vectors
        topo_features = []

        # pd_vectors is list, so we need to deal with each persistence diagram points by "for"
        for batch_index in range(batch):
            if perm_op == "topk":
                if k == -1:
                    raise Exception("please input the k parameter")
                topo_feature = torch.topk(w_vec[batch_index], k=k, dim=0, largest=True).values[-1, :]
            elif perm_op == "sum":
                topo_feature = torch.sum(w_vec[batch_index], dim=0, keepdims=False)
            elif perm_op == "max":
                topo_feature = torch.max(w_vec[batch_index], dim=0, keepdims=False)[0]
            elif perm_op == "min":
                topo_feature = torch.min(w_vec[batch_index], dim=0, keepdims=False)[0]
            elif perm_op == "mean":
                topo_feature = torch.mean(w_vec[batch_index], dim=0, keepdims=False)
            else:
                raise Exception("perm_op should be topk, sum, max, min or mean")
            topo_features.append(topo_feature.unsqueeze(0))
        return torch.cat(topo_features, dim=0).view(batch, 1, dim)    

    def forward(self, pointcloud, pd):
        """
        :param pointcloud: point cloud, [B, N, C], where B is batch number of point cloud data, N is the points number, C is dim of feature, usually C is 3
        :return:
        """
        B, N, C = pointcloud.shape
        topology_feature = []

        # PD data
        h1_pd_points, h2_pd_points = point_to_pd(pd)

        if len(self.permutation_equivariant_layers) != 0:
            for i in range(len(self.permutation_equivariant_layers)):
                # PD -> vector
                h1_vec = self.permutation_equivariant_layers[i](h1_pd_points)
                h2_vec = self.permutation_equivariant_layers[i](h2_pd_points)
                # vector -> w * vector
                h1_w_vec = self.permutation_equivariant_weight_layers[i](h1_vec, h1_pd_points)
                h2_w_vec = self.permutation_equivariant_weight_layers[i](h2_vec, h2_pd_points)
                # w_vec -> opt(w_vec)(remark topology_feature)
                topology_h1_feature = self.operation_vector(h1_w_vec, *self.PermutationEquivariantLayer_parameters["op_parameters"]["h1_op"][i])
                topology_h2_feature = self.operation_vector(h2_w_vec, *self.PermutationEquivariantLayer_parameters["op_parameters"]["h2_op"][i])
                # concate represent vectors
                topology_feature.append(torch.cat((topology_h1_feature, topology_h2_feature), 2))

        if len(self.exponential_layers) != 0:
            for i in range(len(self.exponential_layers)):
                # PD -> vector
                h1_vec = self.exponential_layers[i](h1_pd_points)
                h2_vec = self.exponential_layers[i](h2_pd_points)
                # vector -> w * vector
                h1_w_vec = self.exponential_weight_layers[i](h1_vec, h1_pd_points)
                h2_w_vec = self.exponential_weight_layers[i](h2_vec, h2_pd_points)
                # w_vec -> opt(w_vec)(remark topology_feature)
                topology_h1_feature = self.operation_vector(h1_w_vec, *self.ExponentialLayer_parameters["op_parameters"]["h1_op"][i])
                topology_h2_feature = self.operation_vector(h2_w_vec, *self.ExponentialLayer_parameters["op_parameters"]["h2_op"][i])
                # concate represent vectors
                topology_feature.append(torch.cat((topology_h1_feature, topology_h2_feature), 2))

        if len(self.rational_layers) != 0:
            for i in range(len(self.rational_layers)):
                # PD -> vector
                h1_vec = self.rational_layers[i](h1_pd_points)
                h2_vec = self.rational_layers[i](h2_pd_points)
                # vector -> w * vector
                h1_w_vec = self.rational_weight_layers[i](h1_vec, h1_pd_points)
                h2_w_vec = self.rational_weight_layers[i](h2_vec, h2_pd_points)
                # w_vec -> opt(w_vec)(remark topology_feature)
                topology_h1_feature = self.operation_vector(h1_w_vec, *self.RationalLayer_parameters["op_parameters"]["h1_op"][i])
                topology_h2_feature = self.operation_vector(h2_w_vec, *self.RationalLayer_parameters["op_parameters"]["h2_op"][i])
                # concate represent vectors
                topology_feature.append(torch.cat((topology_h1_feature, topology_h2_feature), 2))

        if len(self.rational_hat_layers) != 0:
            for i in range(len(self.rational_hat_layers)):
                # PD -> vector
                h1_vec = self.rational_hat_layers[i](h1_pd_points)
                h2_vec = self.rational_hat_layers[i](h2_pd_points)
                # vector -> w * vector
                h1_w_vec = self.rational_hat_weight_layers[i](h1_vec, h1_pd_points)
                h2_w_vec = self.rational_hat_weight_layers[i](h2_vec, h2_pd_points)
                # w_vec -> opt(w_vec)(remark topology_feature)
                topology_h1_feature = self.operation_vector(h1_w_vec, *self.RationalHatLayer_parameters["op_parameters"]["h1_op"][i])
                topology_h2_feature = self.operation_vector(h2_w_vec, *self.RationalHatLayer_parameters["op_parameters"]["h2_op"][i])
                # concate represent vectors
                topology_feature.append(torch.cat((topology_h1_feature, topology_h2_feature), 2))

        if len(self.landscape_layers) != 0:
            for i in range(len(self.landscape_layers)):
                # PD -> vector
                h1_vec = self.landscape_layers[i](h1_pd_points)
                h2_vec = self.landscape_layers[i](h2_pd_points)
                # vector -> w * vector
                h1_w_vec = self.landscape_weight_layers[i](h1_vec, h1_pd_points)
                h2_w_vec = self.landscape_weight_layers[i](h2_vec, h2_pd_points)
                # w_vec -> opt(w_vec)(remark topology_feature)
                topology_h1_feature = self.operation_vector(h1_w_vec, *self.LandscapeLayer_parameters["op_parameters"]["h1_op"][i])
                topology_h2_feature = self.operation_vector(h2_w_vec, *self.LandscapeLayer_parameters["op_parameters"]["h2_op"][i])
                # concate represent vectors
                topology_feature.append(torch.cat((topology_h1_feature, topology_h2_feature), 2))

        if len(self.betti_layers) != 0:
            for i in range(len(self.betti_layers)):
                # PD -> vector
                h1_vec = self.betti_layers[i](h1_pd_points)
                h2_vec = self.betti_layers[i](h2_pd_points)
                # vector -> w * vector
                h1_w_vec = self.betti_weight_layers[i](h1_vec, h1_pd_points)
                h2_w_vec = self.betti_weight_layers[i](h2_vec, h2_pd_points)
                # w_vec -> opt(w_vec)(remark topology_feature)
                topology_h1_feature = self.operation_vector(h1_w_vec, *self.BettiLayer_parameters["op_parameters"]["h1_op"][i])
                topology_h2_feature = self.operation_vector(h2_w_vec, *self.BettiLayer_parameters["op_parameters"]["h2_op"][i])
                # concate represent vectors
                topology_feature.append(torch.cat((topology_h1_feature, topology_h2_feature), 2))

        if len(self.entropy_layers) != 0:
            for i in range(len(self.entropy_layers)):
                # PD -> vector
                h1_vec = self.entropy_layers[i](h1_pd_points)
                h2_vec = self.entropy_layers[i](h2_pd_points)
                # vector -> w * vector
                h1_w_vec = self.entropy_weight_layers[i](h1_vec, h1_pd_points)
                h2_w_vec = self.entropy_weight_layers[i](h2_vec, h2_pd_points)
                # w_vec -> opt(w_vec)(remark topology_feature)
                topology_h1_feature = self.operation_vector(h1_w_vec, *self.EntropyLayer_parameters["op_parameters"]["h1_op"][i])
                topology_h2_feature = self.operation_vector(h2_w_vec, *self.EntropyLayer_parameters["op_parameters"]["h2_op"][i])
                # concate represent vectors
                topology_feature.append(torch.cat((topology_h1_feature, topology_h2_feature), 2))

        if len(self.image_layers) != 0:
            for i in range(len(self.image_layers)):
                # PD -> vector
                h1_vec = self.image_layers[i](h1_pd_points)
                h2_vec = self.image_layers[i](h2_pd_points)
                # vector -> w * vector
                h1_w_vec = self.image_weight_layers[i](h1_vec, h1_pd_points)
                h2_w_vec = self.image_weight_layers[i](h2_vec, h2_pd_points)
                # w_vec -> opt(w_vec)(remark topology_feature)
                topology_h1_feature = self.operation_vector(h1_w_vec, *self.ImageLayer_parameters["op_parameters"]["h1_op"][i])
                topology_h2_feature = self.operation_vector(h2_w_vec, *self.ImageLayer_parameters["op_parameters"]["h2_op"][i])
                # concate represent vectors
                topology_feature.append(torch.cat((topology_h1_feature, topology_h2_feature), 2))

        if len(self.hight_order_guassian_kernel_layers) != 0:
            for i in range(len(self.hight_order_guassian_kernel_layers)):
                # PD -> vector
                h1_vec = self.hight_order_guassian_kernel_layers[i](h1_pd_points)
                h2_vec = self.hight_order_guassian_kernel_layers[i](h2_pd_points)
                # vector -> w * vector
                h1_w_vec = self.hight_order_guassian_kernel_weight_layers[i](h1_vec, h1_pd_points)
                h2_w_vec = self.hight_order_guassian_kernel_weight_layers[i](h2_vec, h2_pd_points)
                # w_vec -> opt(w_vec)(remark topology_feature)
                topology_h1_feature = self.operation_vector(h1_w_vec, *self.HightOrderGuasssianKernel_parameters["op_parameters"]["h1_op"][i])
                topology_h2_feature = self.operation_vector(h2_w_vec, *self.HightOrderGuasssianKernel_parameters["op_parameters"]["h2_op"][i])
                # concate represent vectors
                topology_feature.append(torch.cat((topology_h1_feature, topology_h2_feature), 2))

        if len(self.glp_kernel_layers) != 0:
            for i in range(len(self.glp_kernel_layers)):
                # PD -> vector
                h1_vec = self.glp_kernel_layers[i](h1_pd_points)
                h2_vec = self.glp_kernel_layers[i](h2_pd_points)
                # vector -> w * vector
                h1_w_vec = self.glp_kernel_weight_layers[i](h1_vec, h1_pd_points)
                h2_w_vec = self.glp_kernel_weight_layers[i](h2_vec, h2_pd_points)
                # w_vec -> opt(w_vec)(remark topology_feature)
                topology_h1_feature = self.operation_vector(h1_w_vec, *self.GLPKernel_parameters["op_parameters"]["h1_op"][i])
                topology_h2_feature = self.operation_vector(h2_w_vec, *self.GLPKernel_parameters["op_parameters"]["h2_op"][i])
                # concate represent vectors
                topology_feature.append(torch.cat((topology_h1_feature, topology_h2_feature), 2))

        return torch.concat(topology_feature, 2)


