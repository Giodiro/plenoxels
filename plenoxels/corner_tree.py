from typing import Optional
import torch

from plenoxels.tc_plenoxel import plenoxel_sh_encoder
import plenoxels.c_ext as c_ext
from plenoxels.c_ext import RenderOptions

_corner_tree_max_side = 256


def enc_pos(coo):
    return (coo[:, 0] * (_corner_tree_max_side ** 3) +
            coo[:, 1] * (_corner_tree_max_side ** 2) +
            coo[:, 2] * _corner_tree_max_side).long()


from plenoxels.tc_plenoxel import plenoxel_sh_encoder


class CornerTree(torch.nn.Module):
    def __init__(self,
                 sh_degree: int,
                 init_internal: int,
                 aabb: torch.Tensor,
                 near: float,
                 far: float,
                 init_rgb: float,
                 init_sigma: float,
                 max_samples_per_node: int = 2,
                 max_intersections: int = 512,
                 sigma_thresh: float = 1e-4,
                 stop_thresh: float = 1e-4):
        super().__init__()
        self.sh_degree = sh_degree
        self.data_dim = 3 * (sh_degree + 1) ** 2 + 1
        # TODO: near and far should be transformed
        self.near = near
        self.far = far
        # 1 / diameter
        scaling = 1 / (aabb[1] - aabb[0])
        # 0.5 - center / diameter
        offset = 0.5 - 0.5 * (aabb[0] + aabb[1]) * scaling
        self.num_samples = 128
        self.white_bkgd = True
        self.init_rgb = init_rgb
        self.init_sigma = init_sigma
        self.sh_encoder = plenoxel_sh_encoder(sh_degree)
        self.max_samples_per_node = max_samples_per_node
        self.max_intersections = max_intersections
        self.sigma_thresh = sigma_thresh
        self.stop_thresh = stop_thresh

        child    = torch.empty(init_internal, 2, 2, 2, dtype=torch.long)
        coords   = torch.empty(init_internal, 2, 2, 2, 3)
        nids     = torch.empty(init_internal, 2, 2, 2, 8, dtype=torch.long)
        depths   = torch.empty(init_internal, dtype=torch.long)
        self.data = torch.nn.EmbeddingBag(1, self.data_dim, mode='sum')           # n_data, data_dim

        offsets_3d = torch.tensor([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                                   [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
        self.register_buffer("offset", offset)
        self.register_buffer("scaling", scaling)
        self.register_buffer("aabb", aabb)
        self.register_buffer("offsets_3d", offsets_3d)
        self.register_buffer("child", child)
        self.register_buffer("coords", coords)
        self.register_buffer("nids", nids)
        self.register_buffer("depths", depths)
        self.register_buffer("ucoo", torch.tensor([]))
        self.n_internal = 0

    def trasform_coords(self, coords):
        """From world-coordinates to tree-coordinates (a cube between 0 and 1)"""
        return self.offset + coords * self.scaling

    @torch.no_grad()
    def refine(self, leaves=None, copy_interp_data=False):
        # 1. figure out the coordinates of the leaves. This is non-trivial, likely requires traeversing the tree.
        # 2. split the leaves. Add child, parent, depth.

        # 3. For each new leaf calculate coordinates of its neighbors.
        # 4. encode neighbor coordinates, and append to existing coordinates
        # 5. Unique with inverse.
        # 6. Run whatever is below
        if leaves is None:
            leaves = (self.child[:self.n_internal] < 0).nonzero(as_tuple=False)
        else:
            leaves_valid = (
                (self.child[leaves[:, 0], leaves[:, 1], leaves[:, 2], leaves[:, 3]] < 0) &
                (leaves[:, 0] < self.n_internal)
            )
            leaves = leaves[leaves_valid]
        sel = (leaves[:, 0], leaves[:, 1], leaves[:, 2], leaves[:, 3])

        dev = self.child.device
        n_int = self.n_internal
        n_nodes = self.n_internal * 8 + 1
        n_int_new = leaves.shape[0] if n_int > 0 else 1
        n_nodes_fin = n_nodes + n_int_new * 8
        print(f"{n_int=}, {n_nodes=}, {n_int_new=}, {n_nodes_fin=}")
        if n_int_new + n_int > self.child.shape[0]:
            raise MemoryError(f"Not enough data-space for refinement. "
                               f"Need {n_int_new + n_int}, Have {self.child.shape[0]}")

        leaf_coo = self.coords[sel] if n_int > 0 else torch.tensor([[0.5, 0.5, 0.5]], device=dev)
        depths = self.depths[sel[0]] if n_int > 0 else torch.tensor([0], device=dev)
        self.depths[n_int: n_int + n_int_new] = depths + 1
        # + 1 since it's the new leaf, and +1 to get half-voxel-size
        new_leaf_sizes = (1 / (2 ** (depths + 2))).unsqueeze(-1).unsqueeze(-1)
        del depths
        n_offsets = self.offsets_3d.unsqueeze(0).repeat(n_int_new, 1, 1).to(new_leaf_sizes.device) * new_leaf_sizes  # [nl, 8, 3]
        # Coordinates of new leaf centers
        new_leaf_coo = leaf_coo.unsqueeze(1) + n_offsets  # [nl, 8, 3]
        del leaf_coo
        self.coords[n_int: n_int + n_int_new].copy_(new_leaf_coo.view(-1, 2, 2, 2, 3))
        # From center to corners of new leafs (nl -> 8*nl=nc)
        new_corners = (new_leaf_coo.view(-1, 1, 3) + n_offsets.repeat_interleave(8, dim=0)).view(-1, 3)
        del new_leaf_coo
        # Encoded corner coordinates (of new leafs and of the old tree)
        new_corners_enc = enc_pos(new_corners)
        if n_int > 0:
            corners_enc = torch.cat((self.ucoo[self.nids[:n_int].view(-1)], new_corners_enc))
        else:
            corners_enc = new_corners_enc

        new_u_cor, new_cor_idx = torch.unique(corners_enc, return_inverse=True, sorted=True)
        print(f"Deduped corner coordinates from {corners_enc.shape[0]} to {new_u_cor.shape[0]}")
        del corners_enc

        # Update the tree-data:
        # a. create new tensor,
        # b. copy exact old data into it (with changed indices).
        # c. copy interpolated data from old tree into it,
        new_data = torch.empty(new_u_cor.shape[0], self.data_dim, device=dev)
        new_data[:, -1].fill_(self.init_sigma)
        new_data[:, :-1].fill_(self.init_rgb)
        if n_int > 0:
            new_data.scatter_(
                0,
                torch.searchsorted(new_u_cor, self.ucoo).unsqueeze(-1).expand(-1, self.data_dim),
                self.data.weight
            )
        if copy_interp_data:
            # unique index (https://github.com/pytorch/pytorch/issues/36748)
            # Index of same size as the unique corners, indexes the original array.
            perm = torch.arange(new_cor_idx.size(0), dtype=new_cor_idx.dtype, device=new_cor_idx.device)
            inverse, perm = new_cor_idx.flip([0]), perm.flip([0])
            u_idx = inverse.new_empty(new_u_cor.size(0)).scatter_(0, inverse, perm)
            del inverse, perm
            # We only care about the 'new_corners' (which were added at this iteration)
            num_old_corners = new_cor_idx.size(0) - new_corners_enc.size(0)
            new_corners_uniq_idx = u_idx[u_idx > num_old_corners] - num_old_corners
            scatter_idx = torch.searchsorted(new_u_cor, new_corners_enc[new_corners_uniq_idx]).unsqueeze(-1).expand(-1, self.data_dim)
            scatter_data = self.query(new_corners[new_corners_uniq_idx], normalize=False)[0]
            print("Copying %d interp points" % (scatter_data.shape[0]))
            new_data.scatter_(0, scatter_idx, scatter_data)
        self.data = torch.nn.EmbeddingBag.from_pretrained(new_data, freeze=False, mode='sum', sparse=False)
        self.ucoo = new_u_cor
        self.nids[:n_int + n_int_new] = new_cor_idx.view(-1, 2, 2, 2, 8)
        self.child[n_int: n_int + n_int_new] = -1
        self.child[sel] = torch.arange(n_int, n_int + n_int_new, device=self.child.device, dtype=self.child.dtype) - sel[0]
        self.n_internal = n_int + n_int_new

    @torch.no_grad()
    def set(self, indices, values, normalize=True):
        n = indices.shape[0]
        if normalize:
            indices = self.trasform_coords(indices)
        indices.clamp_(0.0, 1 - 1e-6)
        remain_indices = torch.arange(n, dtype=torch.long, device=indices.device)
        node_ids = torch.zeros(n, dtype=torch.long, device=indices.device)
        while remain_indices.numel():
            indices *= 2
            floor = torch.floor(indices).clamp_max_(1)
            indices -= floor
            sel = (node_ids[remain_indices], *(floor.long().T),)
            deltas = self.child[sel]

            term_mask = deltas < 0  # terminate when nodes with 0 children encountered (leaves).
            term_indices = remain_indices[term_mask]

            term_nids = self.nids[sel][term_mask]
            term_vals = values[term_indices]
            self.data.weight.scatter_(0, term_nids.view(-1).unsqueeze(-1).expand(-1, self.data_dim), term_vals.repeat(8, 1))

            remain_indices = remain_indices[~term_mask]
            node_ids[remain_indices] += deltas
            indices = indices[~term_mask]

    def query(self, indices, normalize=True, fetch_data=True):
        n = indices.shape[0]

        with torch.autograd.no_grad():
            if normalize:
                indices = self.trasform_coords(indices)
            indices.clamp_(0.0, 1 - 1e-6)
            node_ids = torch.zeros(n, dtype=torch.long, device=indices.device)
            remain_indices = torch.arange(n, dtype=torch.long, device=indices.device)
            floor_indices = torch.zeros(n, 3, dtype=torch.float, device=indices.device)
            xy = indices
            while remain_indices.numel():
                xy *= 2
                floor = torch.floor(xy)
                floor.clamp_max_(1)
                xy -= floor
                sel = (node_ids[remain_indices], *(floor.long().T),)
                deltas = self.child[sel]

                term_mask = deltas < 0  # terminate when nodes with 0 children encountered (leaves).
                term_indices = remain_indices[term_mask]

                indices.scatter_(0, term_indices.unsqueeze(-1).repeat(1, 3), xy[term_mask])
                floor_indices.scatter_(0, term_indices.unsqueeze(-1).repeat(1, 3), floor[term_mask])

                remain_indices = remain_indices[~term_mask]
                if not remain_indices.numel():
                    break

                node_ids[remain_indices] += deltas[~term_mask]
                xy = xy[~term_mask]

        xy = indices
        sel = (node_ids, *(floor_indices.long().T),)
        sel_nids = self.nids[sel]  # n, 8
        weights = torch.stack((
            (1 - xy[:, 0]) * (1 - xy[:, 1]) * (1 - xy[:, 2]),
            (1 - xy[:, 0]) * (1 - xy[:, 1]) * (xy[:, 2]),
            (1 - xy[:, 0]) * (xy[:, 1])     * (1 - xy[:, 2]),
            (1 - xy[:, 0]) * (xy[:, 1])     * (xy[:, 2]),
            (xy[:, 0])     * (1 - xy[:, 1]) * (1 - xy[:, 2]),
            (xy[:, 0])     * (1 - xy[:, 1]) * (xy[:, 2]),
            (xy[:, 0])     * (xy[:, 1])     * (1 - xy[:, 2]),
            (xy[:, 0])     * (xy[:, 1])     * (xy[:, 2]),
        ), dim=1)  # n, 8
        if not fetch_data:
            return sel_nids, weights
        return self.data(sel_nids, per_sample_weights=weights), weights

    @torch.no_grad()
    def density_frequency(self):
        # The 'frequency' of density. This is going to be some kind of local derivative of the density, to identify
        # locations where it is more important to have a high resolution.
        # 0. The data should have a value (float?) for each cell in the tree, indicating the frequency in that cell.

        # We cannot sample the 0-1 cube uniformly, since we would over-represent large cells (in which we're not particularly interested).
        # As a first go we can calculate it in all cells.
        leaf_sel = (self.child[:self.n_internal] < 0).nonzero(as_tuple=True)  # n_leaves, 4
        n_leaves = leaf_sel[0].shape[0]
        neigh_sel = self.nids[leaf_sel]
        w1 = torch.tensor([[-1, -1, 1, 1, -1, -1, 1, 1]], dtype=torch.float, device=self.child.device)
        w2 = torch.tensor([[1, 1, 1, 1, -1, -1, -1, -1]], dtype=torch.float, device=self.child.device)
        w3 = torch.tensor([[1, -1, 1, -1, 1, -1, 1, -1]], dtype=torch.float, device=self.child.device)
        gx = self.data(neigh_sel, per_sample_weights=w1.expand(n_leaves, -1))[:, -1].abs()
        gy = self.data(neigh_sel, per_sample_weights=w2.expand(n_leaves, -1))[:, -1].abs()
        gz = self.data(neigh_sel, per_sample_weights=w3.expand(n_leaves, -1))[:, -1].abs()
        g = gx + gy + gz
        return g

    @torch.no_grad()
    def refine_density(self, threshold: Optional[float] = None, quantile: Optional[float] = None, verbose: bool = False):
        if threshold is None and quantile is None:
            raise ValueError("threshold or quantile must be specified")
        if threshold is not None and quantile is not None:
            raise ValueError("Only one of threshold and quantile can be specified")

        density = self.density_frequency()
        all_leaves = (self.child[:self.n_internal] < 0).nonzero(as_tuple=False)  # n_leaves, 4
        if quantile is not None:
            threshold = torch.quantile(density, quantile)
        if verbose:
            qs = torch.quantile(density, torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], device=density.device))
            print(f"Density 10%={qs[0]:.2f} 30%={qs[1]:.2f} 50%={qs[2]:.2f} 70%={qs[3]:.2f} 90%={qs[4]:.2f}")

        high_complexity_leaves = all_leaves[density > threshold]
        self.refine(high_complexity_leaves, copy_interp_data=True)

    @torch.no_grad()
    def sample_proposal(self, rays_o, rays_d, max_samples, use_ext_sample: bool):
        if use_ext_sample:
            fn_name = f"ctree_gen_samples{get_c_template_str(self.data.weight.dtype, 2, self.sh_degree)}"
            fn = getattr(c_ext, fn_name)
            c_out = fn(self.child.to(dtype=torch.int), self.offset, self.scaling, rays_o, rays_d, self.get_opt())
            dts = c_out.ray_steps
            points_valid = dts > 0
            points = c_out.intersection_pos
        else:
            # scale direction
            rays_d.mul_(self.scaling)
            delta_scale = 1 / torch.linalg.norm(rays_d, dim=1, keepdim=True)
            rays_d.mul_(delta_scale)
            step_size = 1 / max_samples

            offsets_pos = (self.aabb[1] - rays_o) / rays_d  # [batch, 3]
            offsets_neg = (self.aabb[0] - rays_o) / rays_d  # [batch, 3]
            offsets_in = torch.minimum(offsets_pos, offsets_neg)  # [batch, 3]
            start = torch.amax(offsets_in, dim=-1, keepdim=True)  # [batch, 1]
    #         start.clamp_(min=self.near, max=self.far)  # [batch, 1]
            steps = torch.arange(max_samples, dtype=torch.float32, device=self.child.device).unsqueeze(0)  # [1, n_intrs]
            steps = steps.repeat(rays_d.shape[0], 1)   # [batch, n_intrs]
            intersections = start + steps * step_size  # [batch, n_intrs]
            dts = torch.diff(intersections, n=1, dim=1).mul(delta_scale)
            intersections = intersections[:, :-1]
            points = rays_o.unsqueeze(1) + intersections.unsqueeze(2) * rays_d.unsqueeze(1)
            points_valid = ((points > self.aabb[0]) & (points < self.aabb[1])).all(-1)
        return points, dts, points_valid

    def forward_fast(self, rays_o, rays_d, targets):
        fn_name = f"ctree_loss_grad{get_c_template_str(self.data.weight.dtype, 2, self.sh_degree)}"
        fn = getattr(c_ext, fn_name)
        return fn(self.data.weight, self.child.to(dtype=torch.int), self.nids.to(dtype=torch.int),
                  self.offset, self.scaling, rays_o, rays_d, targets, self.get_opt())

    def vol_render(self, interp, rays_d, dt):
        batch, nintrs = interp.shape[0], interp.shape[1]
        sh_mult = self.sh_encoder(rays_d)  # [batch, ch/3]
        sh_mult = sh_mult.unsqueeze(1).expand(batch, nintrs, -1).unsqueeze(2)  # [batch, nintrs, 1, ch/3]
        interp_rgb = interp[..., :-1].view(batch, nintrs, 3, sh_mult.shape[-1])  # [batch, nintrs, 3, ch/3]
        rgb = torch.sum(sh_mult * interp_rgb, dim=-1)  # [batch, nintrs, 3]

        sigma = interp[..., -1]  # [batch, n_intrs-1, 1]

        # Volumetric rendering TODO: * 2.6 needs to be computed.
        alpha = 1 - torch.exp(-torch.relu(sigma) * dt)            # alpha: [batch, n_intrs-1]
        cum_light = torch.cat((torch.ones(rgb.shape[0], 1, dtype=rgb.dtype, device=rgb.device),
                               torch.cumprod(1 - alpha[:, :-1] + 1e-10, dim=-1)), dim=-1)  # [batch, n_intrs-1]
        abs_light = alpha * cum_light  # [batch, n_intersections - 1]
        acc_map = abs_light.sum(-1)    # [batch]

        # Accumulated color over the samples, ignoring background
        rgb = torch.sigmoid(rgb)  # [batch, n_intrs-1, 3]
        rgb_map = (abs_light.unsqueeze(-1) * rgb).sum(dim=-2)  # [batch, 3]

        if self.white_bkgd:
            # Including the white background in the final color
            rgb_map = rgb_map + (1. - acc_map.unsqueeze(1))
        return rgb_map#, pts, dt, valid, iweights, interp

    def forward(self, rays_o, rays_d, use_ext: bool, use_ext_sample=True):
        if use_ext:
            return CornerTreeRenderFn.apply(self.data.weight, self, rays_o, rays_d, self.get_opt(), False)
        else:
            pts, dt, valid = self.sample_proposal(rays_o, rays_d, self.num_samples, use_ext_sample=use_ext_sample)
            batch, nintrs = pts.shape[:2]

            interp_masked, iweights = self.query(pts[valid].view(-1, 3), normalize=not use_ext_sample)
            interp = torch.zeros(batch, nintrs, self.data_dim,
                                 dtype=torch.float32, device=interp_masked.device)
            interp.masked_scatter_(valid.unsqueeze(-1), interp_masked)

            return self.vol_render(interp=interp, rays_d=rays_d, dt=dt)

    def get_opt(self):
        bb = 1.0 if self.white_bkgd else 0.0
        return init_render_opt(
            background_brightness=bb, density_softplus=False, rgb_padding=0.0,
            max_samples_per_node=self.max_samples_per_node,
            max_intersections=self.max_intersections, sigma_thresh=self.sigma_thresh,
            stop_thresh=self.stop_thresh)


class QuantizedCornerTree(torch.nn.Module):
    def __init__(self, tree, quantizer):
        super(QuantizedCornerTree, self).__init__()
        self.tree = tree
        self.quantizer = quantizer

    def forward(self, rays_o, rays_d):
        pts, dt, valid = self.tree.sample_proposal(rays_o, rays_d, None, use_ext_sample=True)
        batch, nintrs = pts.shape[:2]
        sel_nids, iweights = self.tree.query(pts[valid].view(-1, 3), normalize=False, fetch_data=False)
        sel_data = self.tree.data.weight[sel_nids.view(-1), :]
        vq_loss, sel_data_quantized, perplexity, _ = self.quantizer(sel_data).view(sel_nids.shape[0], 8, -1)  # N, 8, dim
        sel_data_interp = (sel_data_quantized * iweights.unsqueeze(-1)).sum(1)  # N, dim

        # interp_masked, iweights = self.tree.query(pts[valid].view(-1, 3), normalize=False)
        # loss, interp_quantized, perplexity, _ = self.quantizer(interp_masked)

        interp = torch.zeros(batch, nintrs, self.tree.data_dim,
                             dtype=torch.float32, device=sel_data_interp.device)
        interp.masked_scatter_(valid.unsqueeze(-1), sel_data_interp)
        rgb_out = self.tree.vol_render(interp, rays_d, dt)
        return vq_loss, perplexity, rgb_out


def init_render_opt(background_brightness: float = 1.0,
                    density_softplus: bool = False,
                    rgb_padding: float = 0.0,
                    max_samples_per_node: int = 2,
                    max_intersections: int = 512,
                    sigma_thresh: float = 1e-4,
                    stop_thresh: float = 1e-4) -> RenderOptions:
    opts = RenderOptions()
    opts.background_brightness = background_brightness
    opts.max_samples_per_node = max_samples_per_node
    opts.max_intersections = max_intersections

    opts.density_softplus = density_softplus
    opts.rgb_padding = rgb_padding

    opts.sigma_thresh = sigma_thresh
    opts.stop_thresh = stop_thresh

    # Following are unused
    opts.step_size = 1.0
    opts.format = 1
    opts.basis_dim = 1
    opts.ndc_width = 1
    opts.ndc_height = 1
    opts.ndc_focal = 1.0
    opts.min_comp = 1
    opts.max_comp = 1

    return opts


# noinspection PyMethodOverriding,PyAbstractClass
class CornerTreeRenderFn(torch.autograd.Function):
    @staticmethod
    def dispatch_vol_render(*args, dtype: torch.dtype, branching: int, sh_degree: int):
        """
        The function name depends on template arguments:
        volume_render{dtype}{branching}d{sh_degree}
        """
        fn_name = f"ctree_render{get_c_template_str(dtype, branching, sh_degree)}"
        fn = getattr(c_ext, fn_name)
        return fn(*args)

    @staticmethod
    def dispatch_vol_render_bwd(*args, dtype: torch.dtype, branching: int, sh_degree: int):
        """
        The function name depends on template arguments:
        volume_render_bwd{dtype}{branching}d{sh_degree}
        """
        fn_name = f"ctree_render_bwd{get_c_template_str(dtype, branching, sh_degree)}"
        fn = getattr(c_ext, fn_name)
        return fn(*args)

    @staticmethod
    def forward(ctx,
                data: torch.Tensor,
                tree: CornerTree,
                rays_o: torch.Tensor,
                rays_d: torch.Tensor,
                opt: RenderOptions,
                need_all_out: bool):
        out = CornerTreeRenderFn.dispatch_vol_render(
            data, tree.child.to(dtype=torch.int), tree.nids.to(dtype=torch.int), tree.offset, tree.scaling,
            rays_o, rays_d, opt, dtype=tree.data.weight.dtype, branching=2, sh_degree=tree.sh_degree)
        ctx.save_for_backward(
            out.rays_d_norm, out.intersection_num, out.ray_steps,
            out.interpolated_vals, out.interpolated_n_ids, out.interpolation_weights,
        )
        ctx.tree = tree
        ctx.opt = opt
        #with torch.no_grad():
        #    CornerTreeRenderFn.lastctx = out
        if need_all_out:
            return out
        return out.output_rgb

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.needs_input_grad[0]:
            rays_d_norm, n_intrs, ray_steps, interpolated_vals, interpolated_n_ids, interpolation_weights = ctx.saved_tensors
            tree = ctx.tree
            out = CornerTreeRenderFn.dispatch_vol_render_bwd(
                tree.data.weight, tree.nids.to(dtype=torch.int), rays_d_norm, n_intrs,
                grad_out.contiguous(), interpolated_vals, interpolated_n_ids, interpolation_weights,
                ray_steps, ctx.opt,
                dtype=tree.data.weight.dtype, branching=2, sh_degree=tree.sh_degree
            )
            return out, None, None, None, None, None
        return None, None, None, None, None, None


def get_c_template_str(dtype, branching, sh_degree) -> str:
    if isinstance(dtype, torch.dtype):
        if dtype == torch.float32:
            dts = 'f'
        elif dtype == torch.float64:
            dts = 'd'
        else:
            raise RuntimeError(f"Dtype {dtype} unsupported.")
    elif isinstance(dtype, str):
        if dtype.lower() == 'float32':
            dts = 'f'
        elif dtype.lower() == 'float64':
            dts = 'd'
        else:
            raise RuntimeError(f"Dtype {dtype} unsupported.")
    else:
        raise TypeError(f"Cannot understand datatype {dtype}")
    return f"{dts}{int(branching)}d{int(sh_degree)}"
