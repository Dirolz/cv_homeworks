import functools
import torch
import triton
import triton.language as tl
from triton.testing import Benchmark, do_bench, perf_report


EPS = 1e-5
HIDDEN = 1024


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def ln_fwd(
    x_ptr, w_ptr, b_ptr, y_ptr,
    mean_ptr, rstd_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + pid * N + offs, mask=mask, other=0.0).to(tl.float32)

    mu = tl.sum(x, axis=0) / N
    xc = x - mu

    var = tl.sum(xc * xc, axis=0) / N
    rstd = tl.rsqrt(var + eps)

    xn = xc * rstd

    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    y = xn * w + b

    tl.store(y_ptr + pid * N + offs, y, mask=mask)
    tl.store(mean_ptr + pid, mu)
    tl.store(rstd_ptr + pid, rstd)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def ln_bwd(
    x_ptr, w_ptr,
    dy_ptr,
    dx_ptr, dw_ptr, db_ptr,
    mean_ptr, rstd_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + pid * N + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + pid * N + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    mu = tl.load(mean_ptr + pid)
    rstd = tl.load(rstd_ptr + pid)

    xc = x - mu
    xn = xc * rstd

    dy = dy * w

    s1 = tl.sum(dy, axis=0)
    s2 = tl.sum(dy * xn, axis=0)

    dx = (dy - s1 / N - xn * s2 / N) * rstd

    tl.store(dx_ptr + pid * N + offs, dx, mask=mask)

    tl.atomic_add(dw_ptr + offs, dy * xn, mask=mask)
    tl.atomic_add(db_ptr + offs, dy, mask=mask)


def ln_torch(x, w, b, eps=1e-5):
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    rstd = torch.rsqrt(var + eps)
    xn = (x - mu) * rstd
    return xn * w + b


def ln_fwd_wrapper(x, w, b, eps=1e-5):
    x = x.contiguous()
    M, N = x.shape

    y = torch.empty_like(x)
    mean = torch.empty((M,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((M,), device=x.device, dtype=torch.float32)

    ln_fwd[(M,)](
        x, w, b, y,
        mean, rstd,
        N=N,
        eps=eps,
    )

    return y, (x, w, b, mean, rstd)


def ln_bwd_wrapper(dy, ctx):
    x, w, b, mean, rstd = ctx
    M, N = x.shape

    dx = torch.empty_like(x)
    dw = torch.zeros_like(w, dtype=torch.float32)
    db = torch.zeros_like(b, dtype=torch.float32)

    ln_bwd[(M,)](
        x, w,
        dy,
        dx, dw, db,
        mean, rstd,
        N=N,
    )

    return dx, dw, db


H = 1024


def make(n, device="cuda"):
    M = max(1, n // H)
    x = torch.randn(M, H, device=device, dtype=torch.bfloat16)
    w = torch.randn(H, device=device, dtype=torch.bfloat16)
    b = torch.randn(H, device=device, dtype=torch.bfloat16)
    return x, w, b


PROVIDERS = {
    "triton": lambda x, w, b: ln_fwd_wrapper(x, w, b)[0],
    "torch": lambda x, w, b: ln_torch(x, w, b),
    "compile": torch.compile(lambda x, w, b: ln_torch(x, w, b)),
}


def bytes_accessed(x):
    return (
        x.numel() * 2 +
        x.numel() * 2 +
        x.shape[-1] * 2 +
        x.shape[-1] * 2
    )


@perf_report([
    Benchmark(
        x_names=["n_elements"],
        x_vals=[2**i for i in range(20, 26)],
        line_arg="provider",
        line_vals=list(PROVIDERS.keys()),
        line_names=list(PROVIDERS.keys()),
        styles=[
            ("#1f77b4", "-"),
            ("#ff7f0e", "--"),
            ("#2ca02c", ":"),
        ],
        ylabel="latency (ms)",
        plot_name="layernorm_latency_alt",
        args={},
    )
])
def bench_latency(n_elements, provider):
    x, w, b = make(n_elements)
    fn = functools.partial(PROVIDERS[provider], x, w, b)
    ms, mn, mx = do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    return ms, mn, mx


@perf_report([
    Benchmark(
        x_names=["n_elements"],
        x_vals=[2**i for i in range(20, 26)],
        line_arg="provider",
        line_vals=list(PROVIDERS.keys()),
        line_names=list(PROVIDERS.keys()),
        styles=[
            ("#d62728", "-"),
            ("#9467bd", "--"),
            ("#8c564b", ":"),
        ],
        ylabel="GB/s",
        plot_name="layernorm_bandwidth_alt",
        args={},
    )
])
def bench_bw(n_elements, provider):
    x, w, b = make(n_elements)

    fn = functools.partial(PROVIDERS[provider], x, w, b)
    ms, mn, mx = do_bench(fn, quantiles=[0.5, 0.2, 0.8])

    bytes_total = (
        x.numel() * 2 +
        x.numel() * 2 +
        w.numel() * 2 +
        b.numel() * 2
    )

    gbps = lambda t: (bytes_total * 1e-9) / (t * 1e-3)

    return gbps(ms), gbps(mn), gbps(mx)


if __name__ == "__main__":
    torch.manual_seed(0)

    x = torch.randn(4, 1024, device="cuda", dtype=torch.float32)
    w = torch.randn(1024, device="cuda", dtype=torch.float32)
    b = torch.randn(1024, device="cuda", dtype=torch.float32)

    y_ref = ln_torch(x, w, b)
    y, ctx = ln_fwd_wrapper(x, w, b)

    torch.testing.assert_close(y_ref, y, atol=1e-2, rtol=1e-2)

    ln_bwd_wrapper(torch.randn_like(x), ctx)

    bench_latency.run(print_data=True, show_plots=True)
    bench_bw.run(print_data=True, show_plots=True)