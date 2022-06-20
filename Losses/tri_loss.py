from multiprocessing import reduction
import torch


def _pairwise_distances(embeddings, squared=True):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = torch.matmul(embeddings, embeddings.T)

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = (
        torch.unsqueeze(square_norm, 0)
        - 2.0 * dot_product
        + torch.unsqueeze(square_norm, 1)
    )

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = torch.maximum(distances, torch.zeros(distances.shape).cuda())
    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = torch.equal(distances, torch.zeros(distances.shape))
        distances = distances + mask * 1e-16

        distances = torch.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _cart_by_mat(p, n):
    p_shape = p.shape
    n_shape = n.shape

    p_ = torch.unsqueeze(p, 1)
    n_ = torch.unsqueeze(n, 2)

    p_ = p_.repeat([1, n_shape[1], 1])
    n_ = n_.repeat([1, 1, p_shape[1]])

    p_ = torch.permute(p_, [0, 2, 1])
    n_ = torch.permute(n_, [0, 2, 1])

    return p_, n_


def make_gt_dist_mat(ground):
    gt_idx = ground
    gt_1 = gt_idx.repeat([1, gt_idx.shape[0]])
    gt_2 = torch.permute(gt_1, [1, 0])

    dist_gt = torch.abs(gt_1 - gt_2)
    return dist_gt


# def gather_nd(x, indices):
#     newshape = indices.shape[:-1] + x.shape[indices.shape[-1] :]
#     indices = indices.reshape(-1, x.shape[-1]).tolist()
#     out = torch.cat([x.__getitem__(tuple(i)) for i in indices])
#     return out.reshape(newshape)


def calculate(
    dist_mat, aff_mat, coef_func=torch.nn.Identity, is_coef=True, epsilon=0.1
):

    p_, n_ = _cart_by_mat(aff_mat, aff_mat)
    w = torch.where(p_ - n_ < 0)
    w = torch.stack(w, 1)
    wt = w.T
    w1 = torch.not_equal(wt[0], wt[1]) * torch.not_equal(wt[0], wt[2])
    w1.unsqueeze_(1)
    w = torch.masked_select(w, w1).view(-1, w.shape[-1])
    num_triplets = w.shape[0]
    ap_cord = torch.stack([w.select(dim=1, index=0), w.select(dim=1, index=1)], -1)
    an_cord = torch.stack([w.select(dim=1, index=0), w.select(dim=1, index=2)], -1)

    p = dist_mat[list(ap_cord.T)]
    n = dist_mat[list(an_cord.T)]
    p_aff = aff_mat[list(ap_cord.T)]
    combined = torch.stack([p, n]).T
    labels = torch.tensor([[1.0, 0.0]]).repeat([num_triplets, 1])
    triplet_loss = torch.nn.functional.cross_entropy(
        combined.cuda(), labels.cuda(), reduction="none"
    )
    if is_coef:
        triplet_loss = triplet_loss * coef_func((1.0 + epsilon) / (p_aff + epsilon))
    else:
        triplet_loss = triplet_loss * 1

    return torch.mean(triplet_loss)


def calc_triplet_loss(feat, age, age_max):

    dist = _pairwise_distances(feat)
    aff = make_gt_dist_mat(age / age_max)

    tri_loss = calculate(dist, aff, lambda x: torch.pow(x, 1.0), True, epsilon=0.1)
    return tri_loss


if __name__ == "__main__":
    age = torch.tensor([[1, 3, 6]]).cuda()
    feat = torch.tensor([[1.0, 1.0, 1.0], [2, 2, 2], [5, 5, 5]]).cuda()
    calc_triplet_loss(feat, age, 1)
