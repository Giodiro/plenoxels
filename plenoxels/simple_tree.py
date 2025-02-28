from typing import Optional, Union

import torch
import torch.nn as nn

import plenoxels.c_ext as c_ext
from plenoxels.c_ext import OctreeCppSpecf, OctreeCppSpecd, RenderOptions

OptTensor = Optional[torch.Tensor]


class Octree(nn.Module):
    def __init__(self,
                 max_internal_nodes: int,
                 initial_levels: int,
                 sh_degree: int,
                 render_opt: RenderOptions,
                 branching: int = 2,
                 radius: OptTensor = None,
                 center: OptTensor = None,
                 parent_sum: bool = True,
                 dtype: torch.dtype = torch.float32
                 ):
        super().__init__()

        self.max_internal_nodes = max_internal_nodes
        self.sh_degree = sh_degree
        self.data_dim = 3 * (sh_degree + 1) ** 2 + 1
        self.b = branching
        self.n_internal = 0
        self.max_depth = 0
        self.node_size = self.b ** 3
        self.parent_sum = parent_sum
        self.render_opt = render_opt
        self.data_dt = dtype

        if radius is None:
            radius = torch.tensor([0.5, 0.5, 0.5])
        self.register_buffer("scaling", 0.5 / radius)
        if center is None:
            center = torch.tensor([0.5, 0.5, 0.5])
        self.register_buffer("offset", 0.5 - center * self.scaling)

        data = torch.zeros(self.max_internal_nodes * self.node_size + 1, self.data_dim, dtype=self.data_dt)
        child = torch.zeros(self.max_internal_nodes, self.b, self.b, self.b, dtype=torch.int32)
        is_child_leaf = torch.ones(self.max_internal_nodes, self.b, self.b, self.b, dtype=torch.bool)
        parent = torch.zeros(self.max_internal_nodes * self.node_size + 1, dtype=torch.int32)
        parent[0] = -1  # parent of root node is -1
        depth = torch.zeros(self.max_internal_nodes * self.node_size + 1, dtype=torch.int32)

        self.register_parameter("data", nn.Parameter(data))
        self.register_buffer("child", child)
        self.register_buffer("is_child_leaf", is_child_leaf)
        self.register_buffer("parent", parent)
        self.register_buffer("depth", depth)

        for i in range(initial_levels):
            self.refine()

    @torch.no_grad()
    def refine(self, leaves: OptTensor = None) -> None:
        n_int = self.n_internal
        n_nodes = n_int * self.node_size + 1
        if leaves is None:
            leaves = self.is_child_leaf[:n_int].nonzero(as_tuple=False)
        else:
            leaves_valid = (
                self.is_child_leaf[leaves[:, 0], leaves[:, 1], leaves[:, 2]] &
                (leaves[:, 0] < n_int)
            )
            leaves = leaves[leaves_valid]

        n_int_new = leaves.shape[0]
        if n_int == 0:
            n_int_new = 1
        n_nodes_fin = n_nodes + n_int_new * self.node_size
        if n_int + n_int_new > self.max_internal_nodes:
            raise RuntimeError(f"Cannot refine tree further: maximum capacity "
                               f"({self.max_internal_nodes}) reached.")

        new_child = (
            torch.arange(n_nodes, n_nodes_fin, dtype=torch.int32).view(-1, self.b, self.b, self.b)
            - torch.arange(n_int, n_int + n_int_new).view(-1, 1, 1, 1)
        )
        self.child[n_int: n_int + n_int_new] = new_child
        self.is_child_leaf[leaves[:, 0], leaves[:, 1], leaves[:, 2]] = False
        if n_int == 0:
            self.parent[n_nodes:n_nodes_fin] = 0
            self.depth[n_nodes:n_nodes_fin] = 1
        else:
            packed_leaves = pack_index_3d(leaves, self.b).repeat_interleave(self.node_size) + 1
            self.parent[n_nodes: n_nodes_fin] = packed_leaves
            old_depths = self.depth[packed_leaves]
            self.max_depth = max(self.max_depth, old_depths.max() + 1)
            self.depth[n_nodes: n_nodes_fin] = old_depths + 1

        self.n_internal += n_int_new

    def tree_spec(self) -> Union[OctreeCppSpecf, OctreeCppSpecd]:
        if not hasattr(self, 'tree_spec_'):
            if self.data_dt == torch.float32:
                self.tree_spec_ = OctreeCppSpecf(
                    self.data,
                    self.child,
                    self.is_child_leaf,
                    self.parent,
                    self.depth,
                    self.scaling,
                    self.offset,
                    self.parent_sum,
                )
            elif self.data_dt == torch.float64:
                self.tree_spec_ = OctreeCppSpecd(
                    self.data,
                    self.child,
                    self.is_child_leaf,
                    self.parent,
                    self.depth,
                    self.scaling,
                    self.offset,
                    self.parent_sum,
                )
        return self.tree_spec_

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
        return VolumeRenderFunction.apply(
            self.data, self, rays_o, rays_d, self.render_opt)

    def dispatch_fn(self, *args, fn_name):
        typed_fn_name = f"{fn_name}{get_c_template_str(self.data_dt, self.b, self.sh_degree)}"
        fn = getattr(c_ext, typed_fn_name)
        return fn(*args)

    def set(self, indices, values, update_avg: bool = True) -> None:
        self.dispatch_fn(self.tree_spec(), indices, values, update_avg, fn_name="octree_set")

    def query(self, indices: torch.Tensor) -> torch.Tensor:
        return self.dispatch_fn(self.tree_spec(), indices, fn_name="octree_query")

    def query_interp(self, indices: torch.Tensor) -> torch.Tensor:
        return self.dispatch_fn(self.tree_spec(), indices, fn_name="octree_query_interp")


# noinspection PyMethodOverriding,PyAbstractClass
class VolumeRenderFunction(torch.autograd.Function):
    @staticmethod
    def dispatch_vol_render(*args, dtype: torch.dtype, branching: int, sh_degree: int):
        """
        The function name depends on template arguments:
        volume_render{dtype}{branching}d{sh_degree}
        """
        fn_name = f"volume_render{get_c_template_str(dtype, branching, sh_degree)}"
        fn = getattr(c_ext, fn_name)
        return fn(*args)

    @staticmethod
    def dispatch_vol_render_bwd(*args, dtype: torch.dtype, branching: int, sh_degree: int):
        """
        The function name depends on template arguments:
        volume_render_bwd{dtype}{branching}d{sh_degree}
        """
        fn_name = f"volume_render_bwd{get_c_template_str(dtype, branching, sh_degree)}"
        fn = getattr(c_ext, fn_name)
        return fn(*args)

    @staticmethod
    def forward(ctx,
                data: torch.Tensor,
                tree: Octree,
                rays_o: torch.Tensor,
                rays_d: torch.Tensor,
                opt: RenderOptions):
        out = VolumeRenderFunction.dispatch_vol_render(
            data, tree.tree_spec(), rays_o, rays_d, opt, dtype=tree.data.dtype, branching=tree.b, sh_degree=tree.sh_degree)
        ctx.save_for_backward(
            rays_o, rays_d, out.interpolated_vals, out.interpolated_n_ids, out.interpolation_weights, out.ray_offsets,
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
            out = VolumeRenderFunction.dispatch_vol_render_bwd(
                tree.tree_spec(), rays_o, rays_d, grad_out.contiguous(),
                interpolated_vals, interpolated_n_ids,
                interpolation_weights, ray_offsets, ray_steps,
                ctx.opt, dtype=tree.data.dtype, branching=tree.b, sh_degree=tree.sh_degree
            )
            return out, None, None, None, None
        return None, None, None, None, None


def pack_index_3d(idx_3d: torch.Tensor, branching: int) -> torch.Tensor:
    mul = torch.tensor([branching ** 3, branching ** 2, branching, 1], dtype=idx_3d.dtype, device=idx_3d.device)
    return idx_3d.mul(mul.unsqueeze(0)).sum(-1)


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


def init_render_opt(background_brightness: float = 1.0,
                    density_softplus: bool = False,
                    rgb_padding: float = 0.0) -> RenderOptions:
    opts = RenderOptions()
    opts.background_brightness = background_brightness
    opts.max_samples_per_node = 1

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


def gradcheck():
    torch.manual_seed(42)
    tree = Octree(max_internal_nodes=11, initial_levels=2, sh_degree=1, render_opt=init_render_opt(),
                  branching=2, radius=None, center=None, parent_sum=True, dtype=torch.float64).cuda()
    with torch.no_grad():
        tree.data.copy_(torch.randn_like(tree.data) + 2)
    rays_o = torch.zeros(2, 3).cuda()
    rays_d = torch.abs(torch.randn(2, 3)).cuda()
    rays_d = rays_d / torch.linalg.norm(rays_d, dim=1, keepdim=True)

    torch.autograd.gradcheck(
        lambda d: VolumeRenderFunction.apply(d, tree, rays_o, rays_d, tree.render_opt), inputs=[tree.data])


if __name__ == "__main__":
    gradcheck()
