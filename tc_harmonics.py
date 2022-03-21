# Spherical Harmonic helper functions, borrowed from svox library (Alex Yu)

#  Copyright 2021 PlenOctree Authors.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
from typing import Sequence, Tuple, List

import torch


# noinspection PyAbstractClass,PyMethodOverriding
class SphericalHarmonics(torch.autograd.Function):
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = (
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    )
    C3 = (
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    )
    C4 = (
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    )

    @staticmethod
    def forward(ctx, sh_data, dirs, deg):
        """
        :param ctx:
        :param sh_data:
            Tensor of size [batch, n_intrs, sh_ch]
        :param dirs:
            Tensor of size [batch, 3]
        :param deg:
        :return:
            Tensor of size [batch, n_intrs, 3]
        """
        # sh_data_list = torch.split(sh_data, 3, dim=2)
        out = sh_data[0] * SphericalHarmonics.C0

        if deg > 0:
            x, y, z = dirs[:, 0].view(-1, 1, 1), dirs[:, 1].view(-1, 1, 1), dirs[:, 2].view(-1, 1,
                                                                                            1)
            out = out.sub_(SphericalHarmonics.C1 * y * sh_data[1])
            out = out.add_(SphericalHarmonics.C1 * z * sh_data[2])
            out = out.sub_(SphericalHarmonics.C1 * x * sh_data[3])
            if deg > 1:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                out = out.add_(SphericalHarmonics.C2[0] * xy * sh_data[4])
                out = out.add_(SphericalHarmonics.C2[1] * yz * sh_data[5])
                out = out.add_(SphericalHarmonics.C2[2] * (2 * zz - xx - yy) * sh_data[6])
                out = out.add_(SphericalHarmonics.C2[3] * xz * sh_data[7])
                out = out.add_(SphericalHarmonics.C2[4] * (xx - yy) * sh_data[8])
                if deg > 2:
                    out = out.add_(SphericalHarmonics.C3[0] * y * (3 * xx - yy) * sh_data[9])
                    out = out.add_(SphericalHarmonics.C3[1] * xy * z * sh_data[10])
                    out = out.add_(SphericalHarmonics.C3[2] * y * (4 * zz - xx - yy) * sh_data[11])
                    out = out.add_(
                        SphericalHarmonics.C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh_data[12])
                    out = out.add_(SphericalHarmonics.C3[4] * x * (4 * zz - xx - yy) * sh_data[13])
                    out = out.add_(SphericalHarmonics.C3[5] * z * (xx - yy) * sh_data[14])
                    out = out.add_(SphericalHarmonics.C3[6] * x * (xx - 3 * yy) * sh_data[15])
                    if deg > 3:
                        out = out.add_(SphericalHarmonics.C4[0] * xy * (xx - yy) * sh_data[16])
                        out = out.add_(SphericalHarmonics.C4[1] * yz * (3 * xx - yy) * sh_data[17])
                        out = out.add_(SphericalHarmonics.C4[2] * xy * (7 * zz - 1) * sh_data[18])
                        out = out.add_(SphericalHarmonics.C4[3] * yz * (7 * zz - 3) * sh_data[19])
                        out = out.add_(
                            SphericalHarmonics.C4[4] * (zz * (35 * zz - 30) + 3) * sh_data[20])
                        out = out.add_(SphericalHarmonics.C4[5] * xz * (7 * zz - 3) * sh_data[21])
                        out = out.add_(
                            SphericalHarmonics.C4[6] * (xx - yy) * (7 * zz - 1) * sh_data[22])
                        out = out.add_(SphericalHarmonics.C4[7] * xz * (xx - 3 * yy) * sh_data[23])
                        out = out.add_(
                            SphericalHarmonics.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) *
                            sh_data[24])
        ctx.dirs = dirs
        ctx.deg = deg
        ctx.sh_ch = len(sh_data)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # grad_output: [batch, n_intrs, 3]
        # out: [batch, n_intrs, sh_ch]
        deg = ctx.deg
        sh_ch = ctx.sh_ch
        dirs = ctx.dirs

        out = torch.tile(grad_output, (1, 1, sh_ch))
        olist = out.split(3, 2)

        olist[0].mul_(SphericalHarmonics.C0)
        if deg > 0:
            x, y, z = dirs[:, 0].view(-1, 1, 1), dirs[:, 1].view(-1, 1, 1), dirs[:, 2].view(-1, 1,
                                                                                            1)
            olist[1].mul_((-SphericalHarmonics.C1) * y)
            olist[2].mul_(SphericalHarmonics.C1 * z)
            olist[3].mul_((-SphericalHarmonics.C1) * x)
            if deg > 1:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                olist[4].mul_(SphericalHarmonics.C2[0] * xy)
                olist[5].mul_(SphericalHarmonics.C2[1] * yz)
                olist[6].mul_(SphericalHarmonics.C2[2] * (2 * zz - xx - yy))
                olist[7].mul_(SphericalHarmonics.C2[3] * xz)
                olist[8].mul_(SphericalHarmonics.C2[4] * (xx - yy))
                if deg > 2:
                    olist[9].mul_(SphericalHarmonics.C3[0] * y * (3 * xx - yy))
                    olist[10].mul_(SphericalHarmonics.C3[1] * xy * z)
                    olist[11].mul_(SphericalHarmonics.C3[2] * y * (4 * zz - xx - yy))
                    olist[12].mul_(SphericalHarmonics.C3[3] * z * (2 * zz - 3 * xx - 3 * yy))
                    olist[13].mul_(SphericalHarmonics.C3[4] * x * (4 * zz - xx - yy))
                    olist[14].mul_(SphericalHarmonics.C3[5] * z * (xx - yy))
                    olist[15].mul_(SphericalHarmonics.C3[6] * x * (xx - 3 * yy))
                    if deg > 3:
                        olist[16].mul_(SphericalHarmonics.C4[0] * xy * (xx - yy))
                        olist[17].mul_(SphericalHarmonics.C4[1] * yz * (3 * xx - yy))
                        olist[18].mul_(SphericalHarmonics.C4[2] * xy * (7 * zz - 1))
                        olist[19].mul_(SphericalHarmonics.C4[3] * yz * (7 * zz - 3))
                        olist[20].mul_(SphericalHarmonics.C4[4] * (zz * (35 * zz - 30) + 3))
                        olist[21].mul_(SphericalHarmonics.C4[5] * xz * (7 * zz - 3))
                        olist[22].mul_(SphericalHarmonics.C4[6] * (xx - yy) * (7 * zz - 1))
                        olist[23].mul_(SphericalHarmonics.C4[7] * xz * (xx - 3 * yy))
                        olist[24].mul_(
                            SphericalHarmonics.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)))
        return olist, None, None

    @staticmethod
    def test_autograd():
        sh_data = torch.randn(5, 9, 25 * 3).to(dtype=torch.float64).requires_grad_()
        dirs = torch.randn(5, 3).to(dtype=torch.float64)
        deg = 4
        torch.autograd.gradcheck(lambda d: SphericalHarmonics.apply(d, dirs, deg),
                                 inputs=sh_data)


if __name__ == "__main__":
    SphericalHarmonics.test_autograd()


@torch.jit.script
def sh_fwd_apply_list(sh_data: List[torch.Tensor], dirs: torch.Tensor, out: torch.Tensor,
                      deg: int) -> torch.Tensor:
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = (
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    )
    C3 = (
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    )
    C4 = (
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    )
    out = out + (sh_data[0] * C0)

    if deg > 0:
        x, y, z = dirs[:, 0].view(-1, 1, 1), dirs[:, 1].view(-1, 1, 1), dirs[:, 2].view(-1, 1, 1)
        out = out \
              - (C1 * y * sh_data[1]) \
              + (C1 * z * sh_data[2]) \
              - (C1 * x * sh_data[3])
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            out = out \
                  + (C2[0] * xy * sh_data[4]) \
                  + (C2[1] * yz * sh_data[5]) \
                  + (C2[2] * (2 * zz - xx - yy) * sh_data[6]) \
                  + (C2[3] * xz * sh_data[7]) \
                  + (C2[4] * (xx - yy) * sh_data[8])
            if deg > 2:
                out = out \
                      + (C3[0] * y * (3 * xx - yy) * sh_data[9]) \
                      + (C3[1] * xy * z * sh_data[10]) \
                      + (C3[2] * y * (4 * zz - xx - yy) * sh_data[11]) \
                      + (C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh_data[12]) \
                      + (C3[4] * x * (4 * zz - xx - yy) * sh_data[13]) \
                      + (C3[5] * z * (xx - yy) * sh_data[14]) \
                      + (C3[6] * x * (xx - 3 * yy) * sh_data[15])
                if deg > 3:
                    out = out \
                          + (C4[0] * xy * (xx - yy) * sh_data[16]) \
                          + (C4[1] * yz * (3 * xx - yy) * sh_data[17]) \
                          + (C4[2] * xy * (7 * zz - 1) * sh_data[18]) \
                          + (C4[3] * yz * (7 * zz - 3) * sh_data[19]) \
                          + (C4[4] * (zz * (35 * zz - 30) + 3) * sh_data[20]) \
                          + (C4[5] * xz * (7 * zz - 3) * sh_data[21]) \
                          + (C4[6] * (xx - yy) * (7 * zz - 1) * sh_data[22]) \
                          + (C4[7] * xz * (xx - 3 * yy) * sh_data[23]) \
                          + (C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh_data[24])
    return out


@torch.jit.script
def sh_bwd_apply_singleinput(grad_output: torch.Tensor, dirs: torch.Tensor, deg: int) -> List[
    torch.Tensor]:
    # grad_output: [batch, n_intrs, 3]
    # out: [batch, n_intrs, sh_ch]
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = (
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    )
    C3 = (
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    )
    C4 = (
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    )
    olist = [grad_output * (C0)]
    if deg > 0:
        x, y, z = dirs[:, 0].view(-1, 1, 1), dirs[:, 1].view(-1, 1, 1), dirs[:, 2].view(-1, 1, 1)
        olist.append(grad_output * ((-C1) * y))
        olist.append(grad_output * (C1 * z))
        olist.append(grad_output * ((-C1) * x))
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            olist.append(grad_output * (C2[0] * xy))
            olist.append(grad_output * (C2[1] * yz))
            olist.append(grad_output * (C2[2] * (2 * zz - xx - yy)))
            olist.append(grad_output * (C2[3] * xz))
            olist.append(grad_output * (C2[4] * (xx - yy)))
            if deg > 2:
                olist.append(grad_output * (C3[0] * y * (3 * xx - yy)))
                olist.append(grad_output * (C3[1] * xy * z))
                olist.append(grad_output * (C3[2] * y * (4 * zz - xx - yy)))
                olist.append(grad_output * (C3[3] * z * (2 * zz - 3 * xx - 3 * yy)))
                olist.append(grad_output * (C3[4] * x * (4 * zz - xx - yy)))
                olist.append(grad_output * (C3[5] * z * (xx - yy)))
                olist.append(grad_output * (C3[6] * x * (xx - 3 * yy)))
                if deg > 3:
                    olist.append(grad_output * (C4[0] * xy * (xx - yy)))
                    olist.append(grad_output * (C4[1] * yz * (3 * xx - yy)))
                    olist.append(grad_output * (C4[2] * xy * (7 * zz - 1)))
                    olist.append(grad_output * (C4[3] * yz * (7 * zz - 3)))
                    olist.append(grad_output * (C4[4] * (zz * (35 * zz - 30) + 3)))
                    olist.append(grad_output * (C4[5] * xz * (7 * zz - 3)))
                    olist.append(grad_output * (C4[6] * (xx - yy) * (7 * zz - 1)))
                    olist.append(grad_output * (C4[7] * xz * (xx - 3 * yy)))
                    olist.append(grad_output * (C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))))
    return olist


@torch.jit.script
def sh_bwd_apply_listinput(grad_output: List[torch.Tensor], dirs: torch.Tensor, deg: int) -> List[
    torch.Tensor]:
    # grad_output: [batch, n_intrs, 3]
    # out: [batch, n_intrs, sh_ch]
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = (
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    )
    C3 = (
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    )
    C4 = (
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    )
    olist = grad_output
    if deg > 0:
        x, y, z = dirs[:, 0].view(-1, 1, 1), dirs[:, 1].view(-1, 1, 1), dirs[:, 2].view(-1, 1, 1)
        olist[1] = (olist[0] * ((-C1) * y))
        olist[2] = (olist[0] * (C1 * z))
        olist[3] = (olist[0] * ((-C1) * x))
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            olist[4] = (olist[0] * (C2[0] * xy))
            olist[5] = (olist[0] * (C2[1] * yz))
            olist[6] = (olist[0] * (C2[2] * (2 * zz - xx - yy)))
            olist[7] = (olist[0] * (C2[3] * xz))
            olist[8] = (olist[0] * (C2[4] * (xx - yy)))
            if deg > 2:
                olist[9] = (olist[0] * (C3[0] * y * (3 * xx - yy)))
                olist[10] = (olist[0] * (C3[1] * xy * z))
                olist[11] = (olist[0] * (C3[2] * y * (4 * zz - xx - yy)))
                olist[12] = (olist[0] * (C3[3] * z * (2 * zz - 3 * xx - 3 * yy)))
                olist[13] = (olist[0] * (C3[4] * x * (4 * zz - xx - yy)))
                olist[14] = (olist[0] * (C3[5] * z * (xx - yy)))
                olist[15] = (olist[0] * (C3[6] * x * (xx - 3 * yy)))
                if deg > 3:
                    olist[16] = (olist[0] * (C4[0] * xy * (xx - yy)))
                    olist[17] = (olist[0] * (C4[1] * yz * (3 * xx - yy)))
                    olist[18] = (olist[0] * (C4[2] * xy * (7 * zz - 1)))
                    olist[19] = (olist[0] * (C4[3] * yz * (7 * zz - 3)))
                    olist[20] = (olist[0] * (C4[4] * (zz * (35 * zz - 30) + 3)))
                    olist[21] = (olist[0] * (C4[5] * xz * (7 * zz - 3)))
                    olist[22] = (olist[0] * (C4[6] * (xx - yy) * (7 * zz - 1)))
                    olist[23] = (olist[0] * (C4[7] * xz * (xx - 3 * yy)))
                    olist[24] = (olist[0] * (C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))))
    olist[0].mul_(C0)
    return olist


@torch.jit.script
def eval_sh(deg: int,
            sh: torch.Tensor,  # [batch, n_intersections, sh_channels]
            dirs: torch.Tensor,  # [batch, 3]
            ) -> torch.Tensor:  # [batch, n_intersections, 3]
    """
    sh_channels is (deg + 1) ** 2 time 3
    """
    assert 4 >= deg >= 0
    assert sh.shape[-1] == ((deg + 1) ** 2) * 3
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = (
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    )
    C3 = (
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    )
    C4 = (
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    )

    result = C0 * sh.narrow(2, 0, 3)  # batch, n_intersections, 3
    if deg > 0:
        x, y, z = dirs[:, 0].view(-1, 1, 1), dirs[:, 1].view(-1, 1, 1), dirs[:, 2].view(-1, 1, 1)

        result = (result -
                  C1 * y * sh.narrow(2, 3, 3) +
                  C1 * z * sh.narrow(2, 6, 3) -
                  C1 * x * sh.narrow(2, 9, 3))
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                      C2[0] * xy * sh.narrow(2, 12, 3) +
                      C2[1] * yz * sh.narrow(2, 15, 3) +
                      C2[2] * (2.0 * zz - xx - yy) * sh.narrow(2, 18, 3) +
                      C2[3] * xz * sh.narrow(2, 21, 3) +
                      C2[4] * (xx - yy) * sh.narrow(2, 24, 3))
            if deg > 2:
                result = (result +
                          C3[0] * y * (3 * xx - yy) * sh.narrow(2, 27, 3) +
                          C3[1] * xy * z * sh.narrow(2, 30, 3) +
                          C3[2] * y * (4 * zz - xx - yy) * sh.narrow(2, 33, 3) +
                          C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh.narrow(2, 36, 3) +
                          C3[4] * x * (4 * zz - xx - yy) * sh.narrow(2, 39, 3) +
                          C3[5] * z * (xx - yy) * sh.narrow(2, 42, 3) +
                          C3[6] * x * (xx - 3 * yy) * sh.narrow(2, 45, 3))
                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh.narrow(2, 48, 3) +
                              C4[1] * yz * (3 * xx - yy) * sh.narrow(2, 51, 3) +
                              C4[2] * xy * (7 * zz - 1) * sh.narrow(2, 54, 3) +
                              C4[3] * yz * (7 * zz - 3) * sh.narrow(2, 57, 3) +
                              C4[4] * (zz * (35 * zz - 30) + 3) * sh.narrow(2, 60, 3) +
                              C4[5] * xz * (7 * zz - 3) * sh.narrow(2, 63, 3) +
                              C4[6] * (xx - yy) * (7 * zz - 1) * sh.narrow(2, 66, 3) +
                              C4[7] * xz * (xx - 3 * yy) * sh.narrow(2, 69, 3) +
                              C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh.narrow(2, 72,
                                                                                            3))
    return result


@torch.jit.script
def eval_sh_bases(deg: int, dirs: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.
    :param deg: int SH max degree. Currently, 0-4 supported
    :param dirs: torch.Tensor (..., 3) unit directions
    :return: torch.Tensor (..., (deg+1) ** 2)
    """
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = (
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    )
    C3 = (
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    )
    C4 = (
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    )
    assert 4 >= deg >= 0
    # result = torch.empty((*dirs.shape[:-1], (deg + 1) ** 2), dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = C0
    if deg > 0:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -C1 * y
        result[..., 2] = C1 * z
        result[..., 3] = -C1 * x
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = C2[0] * xy
            result[..., 5] = C2[1] * yz
            result[..., 6] = C2[2] * (2.0 * zz - xx - yy)
            result[..., 7] = C2[3] * xz
            result[..., 8] = C2[4] * (xx - yy)

            if deg > 2:
                result[..., 9] = C3[0] * y * (3 * xx - yy)
                result[..., 10] = C3[1] * xy * z
                result[..., 11] = C3[2] * y * (4 * zz - xx - yy)
                result[..., 12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                result[..., 13] = C3[4] * x * (4 * zz - xx - yy)
                result[..., 14] = C3[5] * z * (xx - yy)
                result[..., 15] = C3[6] * x * (xx - 3 * yy)

                if deg > 3:
                    result[..., 16] = C4[0] * xy * (xx - yy)
                    result[..., 17] = C4[1] * yz * (3 * xx - yy)
                    result[..., 18] = C4[2] * xy * (7 * zz - 1)
                    result[..., 19] = C4[3] * yz * (7 * zz - 3)
                    result[..., 20] = C4[4] * (zz * (35 * zz - 30) + 3)
                    result[..., 21] = C4[5] * xz * (7 * zz - 3)
                    result[..., 22] = C4[6] * (xx - yy) * (7 * zz - 1)
                    result[..., 23] = C4[7] * xz * (xx - 3 * yy)
                    result[..., 24] = C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
    return result
