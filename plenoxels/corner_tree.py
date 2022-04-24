import torch

from plenoxels.tc_plenoxel import plenoxel_sh_encoder
import plenoxels.c_ext as c_ext
from plenoxels.c_ext import RenderOptions

_corner_tree_max_side = 256


def enc_pos(coo):
    return (coo[:, 0] * (_corner_tree_max_side ** 3) +
            coo[:, 1] * (_corner_tree_max_side ** 2) +
            coo[:, 2] * _corner_tree_max_side).long()


class CornerTree(torch.nn.Module):
    def __init__(self,
                 sh_degree: int,
                 init_internal: int,
                 aabb: torch.Tensor,
                 near: float,
                 far: float,
                 init_rgb: float,
                 init_sigma: float):
        super().__init__()
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

        child = torch.empty(init_internal, 2, 2, 2, dtype=torch.long)
        coords = torch.empty(init_internal, 2, 2, 2, 3)
        nids = torch.empty(init_internal, 2, 2, 2, 8, dtype=torch.long)
        depths = torch.empty(init_internal, dtype=torch.long)
        is_child_leaf = torch.ones(init_internal, 2, 2, 2, dtype=torch.bool)
        self.data = torch.nn.EmbeddingBag(1, self.data_dim, mode='sum')  # n_data, data_dim

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
        self.register_buffer("is_child_leaf", is_child_leaf)
        self.register_buffer("ucoo", torch.tensor([]))
        self.n_internal = 0

    def trasform_coords(self, coords):
        """From world-coordinates to tree-coordinates (a cube between 0 and 1)"""
        return self.offset + coords * self.scaling

    @torch.no_grad()
    def refine(self, leaves=None):
        # 1. figure out the coordinates of the leaves. This is non-trivial, likely requires traeversing the tree.
        # 2. split the leaves. Add child, is_child_leaf, parent, depth.

        # 3. For each new leaf calculate coordinates of its neighbors.
        # 4. encode neighbor coordinates, and append to existing coordinates
        # 5. Unique with inverse.
        # 6. Run whatever is below
        if leaves is None:
            leaves = self.is_child_leaf[:self.n_internal].nonzero(as_tuple=False)  # n_leaves, 4
        sel = (leaves[:, 0], leaves[:, 1], leaves[:, 2], leaves[:, 3])

        n_int = self.n_internal
        n_nodes = self.n_internal * 8 + 1
        n_int_new = leaves.shape[0] if n_int > 0 else 1
        n_nodes_fin = n_nodes + n_int_new * 8
        print(f"{n_int=}, {n_nodes=}, {n_int_new=}, {n_nodes_fin=}")
        if n_int_new + n_int > self.child.shape[0]:
            raise RuntimeError(f"Not enough data-space for refinement. "
                               f"Need {n_int_new + n_int}, Have {self.child.shape[0]}")

        leaf_coo = self.coords[sel] if n_int > 0 else torch.tensor([[0.5, 0.5, 0.5]])
        depths = self.depths[sel[0]] if n_int > 0 else torch.tensor([0])
        # + 1 since it's the new leaf, and +1 to get half-voxel-size
        new_leaf_sizes = (1 / (2 ** (depths + 2))).unsqueeze(-1).unsqueeze(-1)

        n_offsets = self.offsets_3d.unsqueeze(0).repeat(n_int_new, 1,
                                                        1) * new_leaf_sizes  # [nl, 8, 3]
        new_child = (
                torch.arange(n_nodes, n_nodes_fin, dtype=torch.int32).view(-1, 2, 2, 2)
                - torch.arange(n_int, n_int + n_int_new).view(-1, 1, 1, 1)
        )
        self.child[n_int: n_int + n_int_new] = new_child
        self.is_child_leaf[sel] = False

        new_leaf_coo = leaf_coo.unsqueeze(1) + n_offsets  # [nl, 8, 3]
        self.coords[n_int: n_int + n_int_new] = new_leaf_coo.view(-1, 2, 2, 2, 3)
        self.depths[n_int: n_int + n_int_new] = depths + 1

        self.n_internal = n_int + n_int_new

        # From leaf center to corners (nl -> 8*nl=nc)
        new_corners = (new_leaf_coo.view(-1, 1, 3) + n_offsets.repeat_interleave(8, dim=0)).view(-1,
                                                                                                 3)
        new_corners_enc = enc_pos(new_corners)
        # Need to get all the encoded corner positions of the whole tree.
        if n_int > 0:
            corners_enc = torch.cat((self.ucoo[self.nids[:n_int].view(-1)], new_corners_enc))
        else:
            corners_enc = new_corners_enc

        new_u_cor, new_cor_idx = torch.unique(corners_enc, return_inverse=True, sorted=True)
        print(f"Deduped corner coordinates from {corners_enc.shape[0]} to {new_u_cor.shape[0]}")

        # Update the tree-data: create new tensor, copy the old data into it (with changed indices).
        new_data = torch.zeros(new_u_cor.shape[0], self.data_dim)
        new_data[:, :-1].fill_(self.init_rgb)
        new_data[:, -1].fill_(self.init_sigma)
        if n_int > 0:
            new_data[torch.searchsorted(new_u_cor, self.ucoo), :] = self.data.weight
        self.data = torch.nn.EmbeddingBag.from_pretrained(new_data, freeze=False, mode='sum',
                                                          sparse=False)
        self.ucoo = new_u_cor
        # TODO: parent data should be copied into the corresponding children

        # New neighbor indices
        self.nids[:n_int + n_int_new] = new_cor_idx.view(-1, 2, 2, 2, 8)

    def query(self, indices):
        n = indices.shape[0]

        with torch.autograd.no_grad():
            indices = self.trasform_coords(indices)
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

                term_mask = self.is_child_leaf[
                    sel]  # terminate when nodes with 0 children encountered (leaves).
                term_indices = remain_indices[term_mask]

                indices.scatter_(0, term_indices.unsqueeze(-1).repeat(1, 3), xy[term_mask])
                floor_indices.scatter_(0, term_indices.unsqueeze(-1).repeat(1, 3), floor[term_mask])

                remain_indices = remain_indices[~term_mask]
                if not remain_indices.numel():
                    break

                node_ids[remain_indices] += deltas
                xy = xy[~term_mask]

        xy = indices
        sel = (node_ids, *(floor_indices.long().T),)
        sel_nids = self.nids[sel]  # n, 8
        weights = torch.stack((
            (1 - xy[:, 0]) * (1 - xy[:, 1]) * (1 - xy[:, 2]),
            (1 - xy[:, 0]) * (1 - xy[:, 1]) * (xy[:, 2]),
            (1 - xy[:, 0]) * (xy[:, 1]) * (1 - xy[:, 2]),
            (1 - xy[:, 0]) * (xy[:, 1]) * (xy[:, 2]),
            (xy[:, 0]) * (1 - xy[:, 1]) * (1 - xy[:, 2]),
            (xy[:, 0]) * (1 - xy[:, 1]) * (xy[:, 2]),
            (xy[:, 0]) * (xy[:, 1]) * (1 - xy[:, 2]),
            (xy[:, 0]) * (xy[:, 1]) * (xy[:, 2]),
        ), dim=1)  # n, 8
        return self.data(sel_nids, per_sample_weights=weights)

    @torch.no_grad()
    def sample_proposal(self, rays_o, rays_d, max_samples):
        #         rays_o = self.trasform_coords(rays_o)
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
        steps = torch.arange(max_samples, dtype=torch.float32, device=self.child.device).unsqueeze(
            0)  # [1, n_intrs]
        steps = steps.repeat(rays_d.shape[0], 1)  # [batch, n_intrs]
        intersections = start + steps * step_size  # [batch, n_intrs]
        dts = torch.diff(intersections, n=1, dim=1).mul(delta_scale)
        intersections = intersections[:, :-1]
        points = rays_o.unsqueeze(1) + intersections.unsqueeze(2) * rays_d.unsqueeze(1)
        points_valid = ((points > self.aabb[0]) & (points < self.aabb[1])).all(-1)
        return points, dts, points_valid

    def forward(self, rays_o, rays_d, use_ext: bool):
        if use_ext:
            bb = 1.0 if self.white_bkgd else 0.0
            opt = init_render_opt(background_brightness=bb, density_softplus=False, rgb_padding=0.0)
            return CornerTreeRenderFn.apply(self.data.weight, self, rays_o, rays_d, opt)
        else:
            # NOTE: sample_proposal modifies rays_d.
            pts, dt, valid = self.sample_proposal(rays_o, rays_d, self.num_samples)
            batch, nintrs = pts.shape[:2]

            interp_masked = self.query(pts[valid].view(-1, 3))
            interp = torch.zeros(batch, nintrs, self.data_dim,
                                 dtype=torch.float32, device=interp_masked.device)
            interp.masked_scatter_(valid.unsqueeze(-1), interp_masked)

            sh_mult = self.sh_encoder(rays_d)  # [batch, ch/3]
            sh_mult = sh_mult.unsqueeze(1).expand(batch, nintrs, -1).unsqueeze(
                2)  # [batch, nintrs, 1, ch/3]
            interp_rgb = interp[..., :-1].view(batch, nintrs, 3,
                                               sh_mult.shape[-1])  # [batch, nintrs, 3, ch/3]
            rgb = torch.sum(sh_mult * interp_rgb, dim=-1)  # [batch, nintrs, 3]

            sigma = interp[..., -1]  # [batch, n_intrs-1, 1]

            # Volumetric rendering
            alpha = 1 - torch.exp(-torch.relu(sigma) * dt)  # alpha: [batch, n_intrs-1]
            cum_light = torch.cat((torch.ones(rgb.shape[0], 1, dtype=rgb.dtype, device=rgb.device),
                                   torch.cumprod(1 - alpha[:, :-1] + 1e-10, dim=-1)),
                                  dim=-1)  # [batch, n_intrs-1]
            abs_light = alpha * cum_light  # [batch, n_intersections - 1]
            acc_map = abs_light.sum(-1)  # [batch]

            # Accumulated color over the samples, ignoring background
            rgb = torch.sigmoid(rgb)  # [batch, n_intrs-1, 3]
            rgb_map = (abs_light.unsqueeze(-1) * rgb).sum(dim=-2)  # [batch, 3]

            if self.white_bkgd:
                # Including the white background in the final color
                rgb_map = rgb_map + (1. - acc_map.unsqueeze(1))

            return rgb_map


def init_render_opt(background_brightness: float = 1.0,
                    density_softplus: bool = False,
                    rgb_padding: float = 0.0) -> RenderOptions:
    opts = RenderOptions()
    opts.background_brightness = background_brightness
    opts.max_samples_per_node = 1
    opts.max_intersections = 256

    opts.density_softplus = density_softplus
    opts.rgb_padding = rgb_padding

    opts.sigma_thresh = 1e-2
    opts.stop_thresh = 1e-2

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
                opt: RenderOptions):
        out = CornerTreeRenderFn.dispatch_vol_render(
            data, tree.child, tree.is_child_leaf, tree.nids, tree.offset, tree.scaling,
            rays_o, rays_d, opt, dtype=tree.data.dtype, branching=tree.b, sh_degree=tree.sh_degree)
        ctx.save_for_backward(
            rays_o, rays_d, out.interpolated_vals, out.interpolated_n_ids,
            out.interpolation_weights, out.ray_offsets,
            out.ray_steps
        )
        ctx.tree = tree
        ctx.opt = opt
        return out.output_rgb

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.needs_input_grad[0]:
            rays_o, rays_d, interpolated_vals, interpolated_n_ids, interpolation_weights, ray_offsets, ray_steps = ctx.saved_tensors
            tree = ctx.tree
            out = CornerTreeRenderFn.dispatch_vol_render_bwd(
                tree.data.weight, tree.nids, tree.offset, tree.scaling, rays_o, rays_d,
                grad_out.contiguous(), interpolated_vals, interpolated_n_ids, interpolation_weights,
                ray_offsets, ray_steps, ctx.opt,
                dtype=tree.data.dtype, branching=tree.b, sh_degree=tree.sh_degree
            )
            return out, None, None, None, None
        return None, None, None, None, None


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
