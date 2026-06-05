
import tilelang
import tilelang.language as T
import torch
####################### Norm & int8 #################################################

def _sumsq_maxabs(
    M: int = 4096, N: int=6144, 
    BLOCK_M: int = 64, BLOCK_N: int = 32, 
    threads: int = 256,
    dtype: str = 'float32'
):
    @T.prim_func
    def _sumsq_maxabs_(
        # input
        A: T.Tensor((M, N), dtype), # type: ignore
        # output (init to 0)
        A_max: T.Tensor((1,), T.float32), # type: ignore
        A_sumsq_or_norm: T.Tensor((1,), T.float32) # type: ignore
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=threads) as (pid_n, pid_m):
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
            T.copy(A[pid_m * BLOCK_M, pid_n * BLOCK_N], C_local)
            Cij = T.alloc_var(T.float32)
            Cmax_reducer = T.alloc_reducer((1,), T.float32, op='max', replication="all")
            T.fill(Cmax_reducer, 0)
            Csquare_reducer = T.alloc_reducer((1,), T.float32, op='sum', replication="all")
            T.fill(Csquare_reducer, 0)
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                Cij = T.cast(C_local[i, j], T.float32)
                val1 = T.abs(Cij)
                Cmax_reducer[0] = T.max(val1, Cmax_reducer[0])
                val2 = Cij * Cij
                Csquare_reducer[0] += T.cast(val2, T.float32)
            T.finalize_reducer(Cmax_reducer)
            T.finalize_reducer(Csquare_reducer)
            if T.get_lane_idx() == 0 and T.get_warp_idx() == 0:
                T.atomic_max(A_max[0], Cmax_reducer[0])
                T.atomic_add(A_sumsq_or_norm[0], Csquare_reducer[0])
    return _sumsq_maxabs_

def _scale_int8(
    M: int = 4096, N: int=6144, 
    BLOCK_M: int = 128, BLOCK_N: int = 32, 
    threads: int = 256,
    dtype: str = 'float32',
    dtype2: str = 'float32',
    use_norm: bool = False,
    eps: float = 1e-7,
):
    @T.prim_func
    def _scale_int8_(
        # input
        A: T.Tensor((M, N), dtype), # type: ignore
        A_max: T.Tensor((1,), T.float32), # type: ignore
        A_sumsq_or_norm: T.Tensor((1,), dtype2), # type: ignore
        # output
        A_int8: T.Tensor((M, N), T.int8),  # type: ignore
        A_scale: T.Tensor((1,), T.float32) # type: ignore
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=threads) as (pid_n, pid_m):
            max_A = T.alloc_reducer((1,), dtype, op='max', replication="all")
            max_A[0] = A_max[0]
            if pid_n == 0 and pid_m == 0 and T.get_lane_idx() == 0 and T.get_warp_idx() == 0:
                A_scale[0] = A_max[0] / (127.0 * T.max(
                        T.cast(A_sumsq_or_norm[0], T.float32) 
                        if use_norm else 
                        T.sqrt(T.cast(A_sumsq_or_norm[0], T.float32)), 
                        eps
                    )
                )
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
            C_local_i8 = T.alloc_fragment((BLOCK_M, BLOCK_N), T.int8)
            T.copy(A[pid_m * BLOCK_M, pid_n * BLOCK_N], C_local)
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                C_local_i8[i, j] = T.cast(T.round((127.0 / A_max[0]) * T.cast(C_local[i, j], T.float32)), T.int8)
            T.copy(
                C_local_i8,
                A_int8[pid_m * BLOCK_M, pid_n * BLOCK_N]
            )
    return _scale_int8_


####################### AAT & ATA & completion #################################################
def _aat_int8_max(
    M: int = 4096, K: int = 6144,
    BLOCK_M: int = 128, BLOCK_N: int = 64, BLOCK_K: int = 128,
    threads: int = 128, num_stages: int = 3,
    dtype: str = 'int8', accum_dtype: str = 'int32'
):
    R = BLOCK_M // BLOCK_N
    U = T.ceildiv(M, BLOCK_M)
    total_blocks = (U * (U - 1) // 2) * R + T.ceildiv(M, BLOCK_N)
    @T.prim_func
    def _aat_int8_max_(
        # input
        A: T.Tensor((M, K), dtype), # type: ignore
        # output
        C: T.Tensor((M, M), accum_dtype), # type: ignore
        # output (init to 0) 
        C_max: T.Tensor((1,), accum_dtype) # type: ignore
    ):
        with T.Kernel(total_blocks, threads=threads) as (pid,):
            T.assume(BLOCK_M % BLOCK_N == 0)
            base_pid = pid // R
            pid_m = T.alloc_var(T.int32)
            # pid_n = T.alloc_var(T.int32)
            pid_m = T.cast(T.sqrt(8.0 * T.cast(base_pid, T.float32) + 1.0)*0.5 - 0.5, T.int32)
            pid_m_col = base_pid - (pid_m * (pid_m + 1) // 2)
            pid_n = pid_m_col * R + (pid % R)
            T.assume(pid_m >= 0)
            T.assume(pid_n >= 0)
            T.assume(pid_m < T.ceildiv(M, BLOCK_M))
            T.assume(pid_n < T.ceildiv(M, BLOCK_N))
            # if pid_n >= T.ceildiv(M, BLOCK_N):
            #     T.thread_return()

            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            T.clear(C_local)
            # Compute Tile
            for k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=num_stages):
                T.copy(A[pid_m * BLOCK_M, k * BLOCK_K], A_shared)
                T.copy(A[pid_n * BLOCK_N, k * BLOCK_K], B_shared) 
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            # 1. Write the primary block
            T.copy(C_local, C[pid_m * BLOCK_M, pid_n * BLOCK_N])
            Cmax_reducer = T.alloc_reducer((1,), accum_dtype, op='max', replication="all")
            T.fill(Cmax_reducer, 0)
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                val = T.if_then_else(
                    pid_m * BLOCK_M + i == pid_n * BLOCK_N + j,
                    0,
                    T.abs(C_local[i, j])
                )
                Cmax_reducer[0] = T.max(val, Cmax_reducer[0])
            T.finalize_reducer(Cmax_reducer)
            if T.get_lane_idx() == 0 and T.get_warp_idx() == 0:
                T.atomic_max(C_max[0], Cmax_reducer[0])
    return _aat_int8_max_

def _int32_compl_symm_int8(
    M: int = 4096, 
    BLOCK_M: int = 64, BLOCK_N: int = 32, 
    threads: int = 256
):
    R = BLOCK_M // BLOCK_N
    U = T.ceildiv(M, BLOCK_M)
    total_blocks = (U * (U - 1) // 2) * R + T.ceildiv(M, BLOCK_N)
    @T.prim_func
    def _int32_compl_symm_int8_(
        # input
        A32_lowtri: T.Tensor((M, M), T.int32), # type: ignore
        A_max: T.Tensor((1,), T.int32), # type: ignore
        orig_scale: T.Tensor((1,), T.float32), # type: ignore
        # output 
        A8: T.Tensor((M, M), T.int8),  # type: ignore
        A_scale: T.Tensor((1,), T.float32), # type: ignore
        A_diag: T.Tensor((M,), T.float32),  # type: ignore
    ):
        # T.annotate_restrict_buffers(A_scale, orig_scale)
        with T.Kernel(total_blocks, threads=threads) as (pid,):
            T.assume(BLOCK_M % BLOCK_N == 0)
            base_pid = pid // R
            pid_m = T.alloc_var(T.int32)
            # pid_n = T.alloc_var(T.int32)
            pid_m = T.cast(T.sqrt(8.0 * T.cast(base_pid, T.float32) + 1.0)*0.5 - 0.5, T.int32)
            pid_m_col = base_pid - (pid_m * (pid_m + 1) // 2)
            pid_n = pid_m_col * R + (pid % R)
            T.assume(pid_m >= 0)
            T.assume(pid_n >= 0)
            T.assume(pid_m < T.ceildiv(M, BLOCK_M))
            T.assume(pid_n < T.ceildiv(M, BLOCK_N))

            scale_A = T.alloc_var(T.float32)
            scale_A = 127.0 / T.cast(A_max[0], T.float32)
            A32_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), T.int32)
            A8_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), T.int8)
            A8_shared = T.alloc_shared((BLOCK_M, BLOCK_N), T.int8)
            A8_shared_T = T.alloc_shared((BLOCK_N, BLOCK_M), T.int8)
            T.copy(A32_lowtri[pid_m * BLOCK_M, pid_n * BLOCK_N], A32_frag)
            
            if (pid_n + 1) * BLOCK_N <= pid_m * BLOCK_M: # strictly lower
                for (i, j) in T.Parallel(BLOCK_M, BLOCK_N):
                    A8_frag[i,j] = T.cast(T.round(T.cast(A32_frag[i,j], T.float32) * scale_A), T.int8)
                T.copy(A8_frag, A8_shared)
                T.copy(A8_shared, A8[pid_m * BLOCK_M, pid_n * BLOCK_N])
                T.transpose(A8_shared, A8_shared_T)
                T.copy(A8_shared_T, A8[pid_n * BLOCK_N, pid_m * BLOCK_M])
            else:
                AA_scale = T.alloc_var(T.float32)
                AA_scale = orig_scale[0]
                AA_scale *= AA_scale
                row = T.alloc_var(T.int32)
                col = T.alloc_var(T.int32)
                
                for (i, j) in T.Parallel(BLOCK_M, BLOCK_N):
                    row = pid_m * BLOCK_M + i
                    T.assume(row >= 0)
                    col = pid_n * BLOCK_N + j
                    A8_frag[i,j] = T.cast(
                        T.if_then_else(
                            row == col,
                            0,
                            T.round(T.cast(A32_frag[i,j], T.float32) * scale_A)
                        ), T.int8
                    )
                    if row == col:
                        A_diag[row] = T.cast(A32_frag[i,j], T.float32) * AA_scale

                T.copy(A8_frag, A8[pid_m * BLOCK_M, pid_n * BLOCK_N])

                if pid_n == 0 and pid_m == 0 and T.get_lane_idx() == 0 and T.get_warp_idx() == 0:
                    A_scale[0] = AA_scale / scale_A
    return _int32_compl_symm_int8_

####################### typeII & completion #################################################

def _typeii_int8_sq(
    M: int = 4096,
    BLOCK_M: int = 128, BLOCK_N: int = 64, BLOCK_K: int = 128,
    threads: int = 128, num_stages: int = 3,
    # ALPHA: float = 0.0,
    # BETA: float = 1.0
):
    R = BLOCK_M // BLOCK_N
    U = T.ceildiv(M, BLOCK_M)
    total_blocks = (U * (U - 1) // 2) * R + T.ceildiv(M, BLOCK_N)
    @T.prim_func
    def _typeii_int8_sq_(
        A: T.Tensor((M, M), T.int8),  # type: ignore
        A_scale: T.Tensor((1,), T.float32),  # type: ignore
        A_diag: T.Tensor((M,), T.float32),  # type: ignore
        # B: T.Tensor((M, M), T.int8),
        # B_scale: T.Tensor((1,), T.float32),
        # B_diag: T.Tensor((M,), T.float32),
        C: T.Tensor((M, M), T.float32),  # type: ignore
        C_max: T.Tensor((1,), T.float32),  # type: ignore
        ALPHA: T.float32,
        BETA: T.float32
    ):
        with T.Kernel(total_blocks, threads=threads) as (pid,):
            T.assume(BLOCK_M % BLOCK_N == 0)
            base_pid = pid // R
            pid_m = T.alloc_var(T.int32)
            # pid_n = T.alloc_var(T.int32)
            pid_m = T.cast(T.sqrt(8.0 * T.cast(base_pid, T.float32) + 1.0)*0.5 - 0.5, T.int32)
            pid_m_col = base_pid - (pid_m * (pid_m + 1) // 2)
            pid_n = pid_m_col * R + (pid % R)
            T.assume(pid_m >= 0)
            T.assume(pid_n >= 0)
            T.assume(pid_m < T.ceildiv(M, BLOCK_M))
            T.assume(pid_n < T.ceildiv(M, BLOCK_N))
            # if (pid_m + 1) * BLOCK_M > pid_n * BLOCK_N:
            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), T.int8)
            B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), T.int8)
            row_shared = T.alloc_shared((BLOCK_M,), T.float32)
            col_shared = T.alloc_shared((BLOCK_N,), T.float32)
            
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), T.int32)
            C_float = T.alloc_fragment((BLOCK_M, BLOCK_N), T.float32)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(M, BLOCK_K), num_stages=num_stages):
                T.copy(A[pid_m * BLOCK_M, k * BLOCK_K], A_shared)
                T.copy(A[k * BLOCK_K, pid_n * BLOCK_N], B_shared) 
                T.gemm(A_shared, B_shared, C_local)
            T.copy(A_diag[pid_m * BLOCK_M], row_shared)
            T.copy(A_diag[pid_n * BLOCK_N], col_shared)
            A8 = T.alloc_fragment((BLOCK_M, BLOCK_N), T.int8)
            A_var = T.alloc_var(T.float32)
            A_var = A_scale[0]
            for i in T.Parallel(BLOCK_M):
                row_shared[i] = (row_shared[i] * BETA + ALPHA) * A_var
            for j in T.Parallel(BLOCK_N):
                col_shared[j] = col_shared[j] * A_var * BETA
            A_var *= A_var
            A_var *= BETA
            T.copy(A[pid_m * BLOCK_M, pid_n * BLOCK_N], A8)
            Cmax_reducer = T.alloc_reducer((1,), T.float32, op='max', replication="all")
            T.fill(Cmax_reducer, 0)
            for (i, j) in T.Parallel(BLOCK_M, BLOCK_N):
                C_float[i, j] = (
                    T.cast(A8[i,j], T.float32) * (row_shared[i] + col_shared[j]) 
                    + T.cast(C_local[i,j], T.float32) * A_var
                )
                val = T.if_then_else(
                    pid_m * BLOCK_M + i == pid_n * BLOCK_N + j,
                    0,
                    T.abs(C_float[i, j])
                )
                Cmax_reducer[0] = T.max(val, Cmax_reducer[0])
            T.copy(C_float, C[pid_m * BLOCK_M, pid_n * BLOCK_N])
            T.finalize_reducer(Cmax_reducer)
            if T.get_lane_idx() == 0 and T.get_warp_idx() == 0:
                T.atomic_max(C_max[0], Cmax_reducer[0])
    return _typeii_int8_sq_


def _float32_compl_symm_int8_quad(
    M: int = 4096, 
    BLOCK_M: int = 32, BLOCK_N: int = 16, 
    threads: int = 256,
):
    R = BLOCK_M // BLOCK_N
    U = T.ceildiv(M, BLOCK_M)
    total_blocks = (U * (U - 1) // 2) * R + T.ceildiv(M, BLOCK_N)
    @T.prim_func
    def _float32_compl_symm_int8_quad_(
        # input
        A32_lowtri: T.Tensor((M, M), T.float32),  # type: ignore
        A_max: T.Tensor((1,), T.float32),  # type: ignore
        orig_diag: T.Tensor((M,), T.float32),   # type: ignore
        # output 
        A8: T.Tensor((M, M), T.int8),   # type: ignore
        A_scale: T.Tensor((1,), T.float32),  # type: ignore
        A_diag: T.Tensor((M,), T.float32),  # type: ignore
        C0: T.float32,
        C1: T.float32,
        C2: T.float32,
    ):
        with T.Kernel(total_blocks, threads=threads) as (pid,):
            T.assume(BLOCK_M % BLOCK_N == 0)
            base_pid = pid // R
            pid_m = T.alloc_var(T.int32)
            # pid_n = T.alloc_var(T.int32)
            pid_m = T.cast(T.sqrt(8.0 * T.cast(base_pid, T.float32) + 1.0)*0.5 - 0.5, T.int32)
            pid_m_col = base_pid - (pid_m * (pid_m + 1) // 2)
            pid_n = pid_m_col * R + (pid % R)
            T.assume(pid_m >= 0)
            T.assume(pid_n >= 0)
            T.assume(pid_m < T.ceildiv(M, BLOCK_M))
            T.assume(pid_n < T.ceildiv(M, BLOCK_N))
            scale_A = T.alloc_var(T.float32)
            scale_A = 127.0 / T.cast(A_max[0], T.float32)
            A32_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), T.float32)
            A8_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), T.int8)
            A8_shared = T.alloc_shared((BLOCK_M, BLOCK_N), T.int8)
            A8_shared_T = T.alloc_shared((BLOCK_N, BLOCK_M), T.int8)
            T.copy(A32_lowtri[pid_m * BLOCK_M, pid_n * BLOCK_N], A32_frag)
            
            if (pid_n + 1) * BLOCK_N <= pid_m * BLOCK_M: # strictly lower
                for (i, j) in T.Parallel(BLOCK_M, BLOCK_N):
                    A8_frag[i,j] = T.cast(T.round(T.cast(A32_frag[i,j], T.float32) * scale_A), T.int8)
                T.copy(A8_frag, A8_shared)
                T.copy(A8_shared, A8[pid_m * BLOCK_M, pid_n * BLOCK_N])
                T.transpose(A8_shared, A8_shared_T)
                T.copy(A8_shared_T, A8[pid_n * BLOCK_N, pid_m * BLOCK_M])
            else:
                # AA_scale = T.alloc_var(T.float32)
                # AA_scale = orig_scale[0]
                # AA_scale *= AA_scale
                row = T.alloc_var(T.int32)
                col = T.alloc_var(T.int32)
                for (i, j) in T.Parallel(BLOCK_M, BLOCK_N):
                    row = pid_m * BLOCK_M + i
                    T.assume(row >= 0)
                    if M % BLOCK_M == 0:
                        T.assume(row < M)
                    col = pid_n * BLOCK_N + j
                    if M % BLOCK_N == 0:
                        T.assume(col < M)
                    T.assume(col >= 0)
                    A8_frag[i,j] = T.cast(
                        T.if_then_else(
                            row == col,
                            0,
                            T.round(A32_frag[i,j] * scale_A)
                        ), T.int8
                    )
                    if row == col:
                        A_diag[row] = A32_frag[i,j] + (C2 * orig_diag[row] + C1) * orig_diag[row] + C0
                T.copy(A8_frag, A8[pid_m * BLOCK_M, pid_n * BLOCK_N])

                if pid_n == 0 and pid_m == 0 and T.get_lane_idx() == 0 and T.get_warp_idx() == 0:
                    A_scale[0] = 1.0 / scale_A
    return _float32_compl_symm_int8_quad_



#################################################################################################
# int8-Gram only, now useless

def _typeii_int8_ab(
    M: int = 4096,
    BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_K: int = 256,
    threads: int = 128, num_stages: int = 3,
):
    R = BLOCK_M // BLOCK_N
    U = T.ceildiv(M, BLOCK_M)
    total_blocks = (U * (U - 1) // 2) * R + T.ceildiv(M, BLOCK_N)
    @T.prim_func
    def _typeii_int8_ab_(
        A: T.Tensor((M, M), T.int8),   # type: ignore
        A_scale: T.Tensor((1,), T.float32),  # type: ignore
        A_diag: T.Tensor((M,), T.float32),  # type: ignore
        B: T.Tensor((M, M), T.int8),  # type: ignore
        B_scale: T.Tensor((1,), T.float32),  # type: ignore
        B_diag: T.Tensor((M,), T.float32),  # type: ignore
        # out
        C: T.Tensor((M, M), T.float32),  # type: ignore
        # out (to 0)
        C_max: T.Tensor((1,), T.float32),  # type: ignore
    ):
        with T.Kernel(total_blocks, threads=threads) as (pid,):
            T.assume(BLOCK_M % BLOCK_N == 0)
            base_pid = pid // R
            pid_m = T.alloc_var(T.int32)
            # pid_n = T.alloc_var(T.int32)
            pid_m = T.cast(T.sqrt(8.0 * T.cast(base_pid, T.float32) + 1.0)*0.5 - 0.5, T.int32)
            pid_m_col = base_pid - (pid_m * (pid_m + 1) // 2)
            pid_n = pid_m_col * R + (pid % R)
            T.assume(pid_m >= 0)
            T.assume(pid_n >= 0)
            T.assume(pid_m < T.ceildiv(M, BLOCK_M))
            T.assume(pid_n < T.ceildiv(M, BLOCK_N))
            # if (pid_m + 1) * BLOCK_M > pid_n * BLOCK_N:
            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), T.int8)
            B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), T.int8)
            row_shared = T.alloc_shared((BLOCK_M,), T.float32)
            col_shared = T.alloc_shared((BLOCK_N,), T.float32)
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), T.int32)
            C_float = T.alloc_fragment((BLOCK_M, BLOCK_N), T.float32)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(M, BLOCK_K), num_stages=num_stages):
                T.copy(A[pid_m * BLOCK_M, k * BLOCK_K], A_shared)
                T.copy(B[k * BLOCK_K, pid_n * BLOCK_N], B_shared) 
                T.gemm(A_shared, B_shared, C_local)
            T.copy(A_diag[pid_m * BLOCK_M], row_shared)
            T.copy(B_diag[pid_n * BLOCK_N], col_shared)
            A8 = T.alloc_fragment((BLOCK_M, BLOCK_N), T.int8)
            B8 = T.alloc_fragment((BLOCK_M, BLOCK_N), T.int8)
            A_var = T.alloc_var(T.float32)
            A_var = A_scale[0]
            B_var = T.alloc_var(T.float32)
            B_var = B_scale[0]
            T.copy(A[pid_m * BLOCK_M, pid_n * BLOCK_N], A8)
            T.copy(B[pid_m * BLOCK_M, pid_n * BLOCK_N], B8)
            Cmax_reducer = T.alloc_reducer((1,), T.float32, op='max', replication="all")
            T.fill(Cmax_reducer, 0)
            is_diag = T.alloc_var(T.bool)
            ri = T.alloc_var(T.float32)
            cj = T.alloc_var(T.float32)
            for (i, j) in T.Parallel(BLOCK_M, BLOCK_N):
                is_diag = (pid_m * BLOCK_M + i == pid_n * BLOCK_N + j)
                ri = row_shared[i]
                cj = col_shared[j]
                C_float[i, j] = (
                    (A_var * T.cast(A8[i,j], T.float32) + T.if_then_else(is_diag, ri, 0)) * cj 
                    + (ri * T.cast(B8[i,j], T.float32)
                    + T.cast(C_local[i,j], T.float32) * A_var) * B_var
                )
                val = T.if_then_else(is_diag, 0, T.abs(C_float[i, j]))
                Cmax_reducer[0] = T.max(val, Cmax_reducer[0])
            T.copy(C_float, C[pid_m * BLOCK_M, pid_n * BLOCK_N])
            T.finalize_reducer(Cmax_reducer)
            if T.get_lane_idx() == 0 and T.get_warp_idx() == 0:
                T.atomic_max(C_max[0], Cmax_reducer[0])
    return _typeii_int8_ab_

def _float32_ab_to_int8(
    M: int = 4096, 
    BLOCK_M: int = 32, BLOCK_N: int = 16, 
    threads: int = 256,
):
    R = BLOCK_M // BLOCK_N
    U = T.ceildiv(M, BLOCK_M)
    total_blocks = (U * (U - 1) // 2) * R + T.ceildiv(M, BLOCK_N)
    @T.prim_func
    def _float32_ab_to_int8_(
        # input
        A32_lowtri: T.Tensor((M, M), T.float32),  # type: ignore
        A_max: T.Tensor((1,), T.float32),  # type: ignore
        # output 
        A8: T.Tensor((M, M), T.int8),   # type: ignore
        A_scale: T.Tensor((1,), T.float32),  # type: ignore
        A_diag: T.Tensor((M,), T.float32)  # type: ignore
    ):
        with T.Kernel(total_blocks, threads=threads) as (pid,):
            T.assume(BLOCK_M % BLOCK_N == 0)
            base_pid = pid // R
            pid_m = T.alloc_var(T.int32)
            # pid_n = T.alloc_var(T.int32)
            pid_m = T.cast(T.sqrt(8.0 * T.cast(base_pid, T.float32) + 1.0)*0.5 - 0.5, T.int32)
            pid_m_col = base_pid - (pid_m * (pid_m + 1) // 2)
            pid_n = pid_m_col * R + (pid % R)
            T.assume(pid_m >= 0)
            T.assume(pid_n >= 0)
            T.assume(pid_m < T.ceildiv(M, BLOCK_M))
            T.assume(pid_n < T.ceildiv(M, BLOCK_N))
            scale_A = T.alloc_var(T.float32)
            scale_A = 127.0 / T.cast(A_max[0], T.float32)
            A32_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), T.float32)
            A8_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), T.int8)
            A8_shared = T.alloc_shared((BLOCK_M, BLOCK_N), T.int8)
            A8_shared_T = T.alloc_shared((BLOCK_N, BLOCK_M), T.int8)
            T.copy(A32_lowtri[pid_m * BLOCK_M, pid_n * BLOCK_N], A32_frag)
            
            if (pid_n + 1) * BLOCK_N <= pid_m * BLOCK_M: # strictly lower
                for (i, j) in T.Parallel(BLOCK_M, BLOCK_N):
                    A8_frag[i,j] = T.cast(T.round(T.cast(A32_frag[i,j], T.float32) * scale_A), T.int8)
                T.copy(A8_frag, A8_shared)
                T.copy(A8_shared, A8[pid_m * BLOCK_M, pid_n * BLOCK_N])
                T.transpose(A8_shared, A8_shared_T)
                T.copy(A8_shared_T, A8[pid_n * BLOCK_N, pid_m * BLOCK_M])
            else:
                # AA_scale = T.alloc_var(T.float32)
                # AA_scale = orig_scale[0]
                # AA_scale *= AA_scale
                row = T.alloc_var(T.int32)
                col = T.alloc_var(T.int32)
                for (i, j) in T.Parallel(BLOCK_M, BLOCK_N):
                    row = pid_m * BLOCK_M + i
                    T.assume(row >= 0)
                    if M % BLOCK_M == 0:
                        T.assume(row < M)
                    col = pid_n * BLOCK_N + j
                    if M % BLOCK_N == 0:
                        T.assume(col < M)
                    T.assume(col >= 0)
                    A8_frag[i,j] = T.cast(
                        T.if_then_else(
                            row == col,
                            0,
                            T.round(A32_frag[i,j] * scale_A)
                        ), T.int8
                    )
                    if row == col:
                        A_diag[row] = A32_frag[i,j]
                T.copy(A8_frag, A8[pid_m * BLOCK_M, pid_n * BLOCK_N])

                if pid_n == 0 and pid_m == 0 and T.get_lane_idx() == 0 and T.get_warp_idx() == 0:
                    A_scale[0] = 1.0 / scale_A
    return _float32_ab_to_int8_


def _typeii_typei_int8(
    M: int = 4096, N: int= 6144,
    BLOCK_M: int = 128, BLOCK_N: int = 64, BLOCK_K: int = 128,
    threads: int = 128, num_stages: int = 3
):
    @T.prim_func
    def _typeii_typei_int8_(
        A: T.Tensor((M, M), T.int8), # type: ignore
        A_scale: T.Tensor((1,), T.float32), # type: ignore
        A_diag: T.Tensor((M,), T.float32), # type: ignore
        B: T.Tensor((M, N), T.int8), # type: ignore
        B_scale: T.Tensor((1,), T.float32), # type: ignore
        # B_diag: T.TensNor((M,), T.float32),
        C: T.Tensor((M, N), T.float32), # type: ignore
        C_max: T.Tensor((1,), T.float32), # type: ignore
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=threads) as (pid_n, pid_m):
            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), T.int8)
            B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), T.int8)
            row_shared = T.alloc_shared((BLOCK_M,), T.float32)
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), T.int32)
            C_float = T.alloc_fragment((BLOCK_M, BLOCK_N), T.float32)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(M, BLOCK_K), num_stages=num_stages):
                T.copy(A[pid_m * BLOCK_M, k * BLOCK_K], A_shared)
                T.copy(B[k * BLOCK_K, pid_n * BLOCK_N], B_shared) 
                T.gemm(A_shared, B_shared, C_local)
            T.copy(A_diag[pid_m * BLOCK_M], row_shared)
            B8 = T.alloc_fragment((BLOCK_M, BLOCK_N), T.int8)
            A_var = T.alloc_var(T.float32)
            A_var = A_scale[0]
            B_var = T.alloc_var(T.float32)
            B_var = B_scale[0]
            # A_var_beta = A_var * BETA
            T.copy(B[pid_m * BLOCK_M, pid_n * BLOCK_N], B8)
            Cmax_reducer = T.alloc_reducer((1,), T.float32, op='max', replication="all")
            T.fill(Cmax_reducer, 0)
            for (i, j) in T.Parallel(BLOCK_M, BLOCK_N):
                C_float[i, j] = (
                    T.cast(B8[i,j], T.float32) * (row_shared[i]) 
                    + T.cast(C_local[i,j], T.float32) * A_var
                ) * B_var
                val = T.abs(C_float[i, j])
                Cmax_reducer[0] = T.max(val, Cmax_reducer[0])
            T.copy(C_float, C[pid_m * BLOCK_M, pid_n * BLOCK_N])
            T.finalize_reducer(Cmax_reducer)
            if T.get_lane_idx() == 0 and T.get_warp_idx() == 0:
                T.atomic_max(C_max[0], Cmax_reducer[0])
    return _typeii_typei_int8_

def _float32_to_int8(
    M: int = 4096, N: int = 4096, 
    BLOCK_M: int = 32, BLOCK_N: int = 16, 
    threads: int = 128,
):
    @T.prim_func
    def _float32_to_int8_(
        # input
        C32: T.Tensor((M, N), T.float32), # type: ignore
        C_max: T.Tensor((1,), T.float32), # type: ignore
        # output
        C: T.Tensor((M, N), T.int8), # type: ignore
        C_scale: T.Tensor((1,), T.float32), # type: ignore
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=threads) as (pid_n, pid_m):
            scale_C = T.alloc_var(T.float32)
            scale_C = 127.0 / C_max[0]
            C32_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), T.float32)
            C_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), T.int8)
            T.copy(C32[pid_m * BLOCK_M, pid_n * BLOCK_N], C32_frag)
            for (i, j) in T.Parallel(BLOCK_M, BLOCK_N):
                C_frag[i,j] = T.cast(T.round(T.cast(C32_frag[i,j], T.float32) * scale_C), T.int8)
            T.copy(C_frag, C[pid_m * BLOCK_M, pid_n * BLOCK_N]) 
            if pid_n == 0 and pid_m == 0 and T.get_lane_idx() == 0 and T.get_warp_idx() == 0:
                C_scale[0] = 1.0 / scale_C
    return _float32_to_int8_


############### - FP16 - ##################################################################33

def _to_prec(
    M: int = 4096, N: int=6144, 
    BLOCK_M: int = 32, BLOCK_N: int = 16, 
    threads: int = 128,
    dtype: str = 'float32',
    dtype2: str = 'float32',
    dtype_out: str = 'float16',
    use_norm: bool = False,
    eps: float = 1e-7,
):
    @T.prim_func
    def _to_prec_(
        # input
        A: T.Tensor((M, N), dtype), # type: ignore
        A_sumsq_or_norm: T.Tensor((1,), dtype2), # type: ignore
        # output
        A_prec: T.Tensor((M, N), dtype_out),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=threads) as (pid_n, pid_m):
            A_scale = T.alloc_var(T.float32)
            A_scale = 1.0 / T.max(
                T.cast(A_sumsq_or_norm[0], T.float32) if use_norm else T.sqrt(T.cast(A_sumsq_or_norm[0], T.float32)), 
                eps
            )
            A_local = T.alloc_fragment((BLOCK_M, BLOCK_N), T.float32)
            T.copy(A[pid_m * BLOCK_M, pid_n * BLOCK_N], A_local)
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                A_local[i, j] = A_local[i, j] * A_scale
            T.copy(A_local, A_prec[pid_m * BLOCK_M, pid_n * BLOCK_N])
    return _to_prec_


def _ab_prec(M: int, N: int, K: int,
           BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_K: int = 64,
           threads: int = 256, num_stages: int = 3,
           dtype: str = 'float16', accum_dtype: str = 'float32'):
    @T.prim_func
    def _ab_prec_(A: T.Tensor((M, K), dtype),  # type: ignore
               B: T.Tensor((K, N), dtype),  # type: ignore
               C: T.Tensor((M, N), dtype)): # type: ignore
        with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=threads) as (bx, by):
            A_s = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            B_s = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
            C_f = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            T.clear(C_f)

            for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=num_stages):
                T.copy(A[by * BLOCK_M, ko * BLOCK_K], A_s)
                T.copy(B[ko * BLOCK_K, bx * BLOCK_N], B_s)
                T.gemm(A_s, B_s, C_f)

            T.copy(C_f, C[by * BLOCK_M, bx * BLOCK_N])

    return _ab_prec_

def _aat_prec(M: int = 4096, K: int = 6144,
           BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_K: int = 64,
           threads: int = 256, num_stages: int = 3,
           dtype: str = 'float16', accum_dtype: str = 'float32'):

    R = BLOCK_M // BLOCK_N
    U = T.ceildiv(M, BLOCK_M)
    total_blocks = (U * (U - 1) // 2) * R + T.ceildiv(M, BLOCK_N)
    @T.prim_func
    def _aat_prec_(A: T.Tensor((M, K), dtype),  # type: ignore
               C: T.Tensor((M, M), dtype)): # type: ignore
        with T.Kernel(total_blocks, threads=threads) as (pid,):
            T.assume(BLOCK_M % BLOCK_N == 0)
            base_pid = pid // R
            pid_m = T.alloc_var(T.int32)
            # pid_n = T.alloc_var(T.int32)
            pid_m = T.cast(T.sqrt(8.0 * T.cast(base_pid, T.float32) + 1.0)*0.5 - 0.5, T.int32)
            pid_m_col = base_pid - (pid_m * (pid_m + 1) // 2)
            pid_n = pid_m_col * R + (pid % R)
            T.assume(pid_m >= 0)
            T.assume(pid_n >= 0)
            T.assume(pid_m < T.ceildiv(M, BLOCK_M))
            T.assume(pid_n < T.ceildiv(M, BLOCK_N))

            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

            T.clear(C_local)
            
            # Compute Tile
            for k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=num_stages):
                T.copy(A[pid_m * BLOCK_M, k * BLOCK_K], A_shared)
                T.copy(A[pid_n * BLOCK_N, k * BLOCK_K], B_shared) 
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[pid_m * BLOCK_M, pid_n * BLOCK_N])

            s_shared = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
            t_shared = T.alloc_shared((BLOCK_N, BLOCK_M), dtype)
            if (pid_n + 1) * BLOCK_N <= pid_m * BLOCK_M: # strictly lower
                T.copy(C_local, s_shared)
                T.copy(s_shared, C[pid_m * BLOCK_M, pid_n * BLOCK_N])
                T.transpose(s_shared, t_shared)
                T.copy(t_shared, C[pid_n * BLOCK_N, pid_m * BLOCK_M])
            else:
                T.copy(C_local, C[pid_m * BLOCK_M, pid_n * BLOCK_N])

    return _aat_prec_

def _quad_prec(
    M: int = 4096,
    BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_K: int = 64,
    threads: int = 128, num_stages: int = 3,
    dtype: str = 'float16', accum_dtype: str = 'float32'
):
    R = BLOCK_M // BLOCK_N
    U = T.ceildiv(M, BLOCK_M)
    total_blocks = (U * (U - 1) // 2) * R + T.ceildiv(M, BLOCK_N)
    @T.prim_func
    def _quad_prec_(
        A: T.Tensor((M, M), dtype),  # type: ignore
        C: T.Tensor((M, M), dtype),  # type: ignore
        C0: T.float32,
        C1: T.float32,
        C2: T.float32
    ):
        with T.Kernel(total_blocks, threads=threads) as (pid,):
            T.assume(BLOCK_M % BLOCK_N == 0)
            base_pid = pid // R
            pid_m = T.alloc_var(T.int32)
            # pid_n = T.alloc_var(T.int32)
            pid_m = T.cast(T.sqrt(8.0 * T.cast(base_pid, T.float32) + 1.0)*0.5 - 0.5, T.int32)
            pid_m_col = base_pid - (pid_m * (pid_m + 1) // 2)
            pid_n = pid_m_col * R + (pid % R)
            T.assume(pid_m >= 0)
            T.assume(pid_n >= 0)
            T.assume(pid_m < T.ceildiv(M, BLOCK_M))
            T.assume(pid_n < T.ceildiv(M, BLOCK_N))
            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(M, BLOCK_K), num_stages=num_stages):
                T.copy(A[pid_m * BLOCK_M, k * BLOCK_K], A_shared)
                T.copy(A[pid_n * BLOCK_N, k * BLOCK_K], B_shared) 
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            del A_shared
            del B_shared
            s_shared = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
            t_shared = T.alloc_shared((BLOCK_N, BLOCK_M), dtype)
            T.copy(A[pid_m * BLOCK_M, pid_n * BLOCK_N], s_shared)
            if (pid_n + 1) * BLOCK_N <= pid_m * BLOCK_M: # strictly lower
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    s_shared[i, j] = T.cast(
                        C1 * T.cast(s_shared[i, j], T.float32) + C2 * C_local[i, j],
                        dtype
                    )
                T.copy(s_shared, C[pid_m * BLOCK_M, pid_n * BLOCK_N])
                T.transpose(s_shared, t_shared)
                T.copy(t_shared, C[pid_n * BLOCK_N, pid_m * BLOCK_M])
            else:
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    s_shared[i, j] = T.cast(
                        C1 * T.cast(s_shared[i, j], T.float32) + C2 * C_local[i, j] +
                        T.if_then_else((pid_m * BLOCK_M + i == pid_n * BLOCK_N + j), C0, 0),
                        dtype
                    )
                T.copy(s_shared, C[pid_m * BLOCK_M, pid_n * BLOCK_N])
    return _quad_prec_

def _ab_symm_prec(M: int = 4096,
           BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_K: int = 64,
           threads: int = 256, num_stages: int = 3,
           dtype: str = 'float16', accum_dtype: str = 'float32'):

    R = BLOCK_M // BLOCK_N
    U = T.ceildiv(M, BLOCK_M)
    total_blocks = (U * (U - 1) // 2) * R + T.ceildiv(M, BLOCK_N)
    @T.prim_func
    def _ab_symm_prec_(A: T.Tensor((M, M), dtype),  # type: ignore
               B: T.Tensor((M, M), dtype), # type: ignore
               C: T.Tensor((M, M), dtype)): # type: ignore
        with T.Kernel(total_blocks, threads=threads) as (pid,):
            T.assume(BLOCK_M % BLOCK_N == 0)
            base_pid = pid // R
            pid_m = T.alloc_var(T.int32)
            # pid_n = T.alloc_var(T.int32)
            pid_m = T.cast(T.sqrt(8.0 * T.cast(base_pid, T.float32) + 1.0)*0.5 - 0.5, T.int32)
            pid_m_col = base_pid - (pid_m * (pid_m + 1) // 2)
            pid_n = pid_m_col * R + (pid % R)
            T.assume(pid_m >= 0)
            T.assume(pid_n >= 0)
            T.assume(pid_m < T.ceildiv(M, BLOCK_M))
            T.assume(pid_n < T.ceildiv(M, BLOCK_N))

            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

            T.clear(C_local)
            
            for k in T.Pipelined(T.ceildiv(M, BLOCK_K), num_stages=num_stages):
                T.copy(A[pid_m * BLOCK_M, k * BLOCK_K], A_shared)
                T.copy(B[pid_n * BLOCK_N, k * BLOCK_K], B_shared) 
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            del A_shared
            del B_shared

            s_shared = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
            t_shared = T.alloc_shared((BLOCK_N, BLOCK_M), dtype)
            if (pid_n + 1) * BLOCK_N <= pid_m * BLOCK_M: # strictly lower
                T.copy(C_local, s_shared)
                T.copy(s_shared, C[pid_m * BLOCK_M, pid_n * BLOCK_N])
                T.transpose(s_shared, t_shared)
                T.copy(t_shared, C[pid_n * BLOCK_N, pid_m * BLOCK_M])
            else:
                T.copy(C_local, C[pid_m * BLOCK_M, pid_n * BLOCK_N])
                
    return _ab_symm_prec_


BLOCK_Q = 128

############### - Block Quantized FP8 / INT8 - ##################################################################33

def _to_bq(
    M: int = 4096, N: int=6144, 
    threads: int = 128,
    dtype: str = 'float32',
    dtype2: str = 'float32',
    dtype_out: str = 'float8_e4m3fn',
    use_norm: bool = False,
    eps: float = 1e-7,
):
    assert dtype_out in ['float8_e4m3fn', 'float8_e4m3', 'int8']
    md = T.ceildiv(M, BLOCK_Q)
    nd = T.ceildiv(N, BLOCK_Q)
    DTYPE_MAX = 448 if 'float8_e4m3' in dtype_out else 127 
    @T.prim_func
    def _to_bq_(
        # input
        A: T.Tensor((M, N), dtype), # type: ignore
        A_sumsq_or_norm: T.Tensor((1,), dtype2), # type: ignore
        # output
        A_bq: T.Tensor((M, N), dtype_out),  # type: ignore
        A_scale: T.Tensor((md, nd), T.float32) # type: ignore
    ):
        with T.Kernel(nd, md, threads=threads) as (pid_n, pid_m):
            A_scale_1 = T.alloc_var(T.float32)
            A_scale_2 = T.alloc_reducer((1,), T.float32, op="max", replication="all")
            T.fill(A_scale_2, 0)
            A_scale_1 = 1.0 / T.max(
                T.cast(A_sumsq_or_norm[0], T.float32) if use_norm else T.sqrt(T.cast(A_sumsq_or_norm[0], T.float32)), 
                eps
            )
            A_local = T.alloc_fragment((BLOCK_Q, BLOCK_Q), T.float32)
            T.copy(A[pid_m * BLOCK_Q, pid_n * BLOCK_Q], A_local)
            for i, j in T.Parallel(BLOCK_Q, BLOCK_Q):
                A_local[i, j] = A_local[i, j] * A_scale_1
                A_scale_2[0] = T.max(A_scale_2[0], T.abs(A_local[i, j]))
            T.finalize_reducer(A_scale_2)
            A_scale_1 = DTYPE_MAX / A_scale_2[0]
            if (T.get_lane_idx() == 0 and T.get_warp_idx() == 0):
                A_scale[pid_m, pid_n] = 1 / A_scale_1
            for i, j in T.Parallel(BLOCK_Q, BLOCK_Q):
                A_local[i, j] = A_local[i, j] * A_scale_1
            T.copy(A_local, A_bq[pid_m * BLOCK_Q, pid_n * BLOCK_Q])
    return _to_bq_


def _aat_bq(
    M: int = 4096, K: int = 6144,
    threads: int = 128, num_stages: int = 3,
    dtype: str = 'float8_e4m3fn'
):
    md = T.ceildiv(M, BLOCK_Q)
    kd = T.ceildiv(K, BLOCK_Q)
    total_blocks = md * (md + 1) // 2

    DTYPE_MAX = 448 if 'float8_e4m3' in dtype else 127 
    accum_dtype = 'float32' if 'float8_e4m3' in dtype else 'int32'

    @T.prim_func
    def _aat_bq_(
        # input
        A: T.Tensor((M, K), dtype), # type: ignore
        A_scale: T.Tensor((md, kd), 'float32'), # type: ignore
        # output
        C: T.Tensor((M, M), dtype), # type: ignore
        # output (init to 0) 
        C_scale: T.Tensor((md, md), 'float32'), # type: ignore
        C_diag: T.Tensor((M,), 'float32'), # type: ignore
    ):
        with T.Kernel(total_blocks, threads=threads) as (pid,):
            pid_m = T.alloc_var(T.int32)
            pid_m = T.cast(T.sqrt(8.0 * T.cast(pid, T.float32) + 1.0)*0.5 - 0.5, T.int32)
            pid_n = pid - (pid_m * (pid_m + 1) // 2)
            T.assume(pid_m >= 0)
            T.assume(pid_n >= 0)
            T.assume(pid_m < T.ceildiv(M, BLOCK_Q))
            T.assume(pid_n < T.ceildiv(M, BLOCK_Q))
            T.assume(md == T.ceildiv(M, BLOCK_Q))
            T.assume(kd == T.ceildiv(K, BLOCK_Q))
            # if pid_n >= T.ceildiv(M, BLOCK_N):
            #     T.thread_return()
            A_shared = T.alloc_shared((BLOCK_Q, BLOCK_Q), dtype)
            B_shared = T.alloc_shared((BLOCK_Q, BLOCK_Q), dtype)
            C_local = T.alloc_fragment((BLOCK_Q, BLOCK_Q), accum_dtype)
            C_float = T.alloc_fragment((BLOCK_Q, BLOCK_Q), T.float32)
            A_scale_1 = T.alloc_var(T.float32)
            T.clear(C_float)
            # Compute Tile
            for k in T.Pipelined(kd, num_stages=num_stages):
                T.clear(C_local)
                T.copy(A[pid_m * BLOCK_Q, k * BLOCK_Q], A_shared)
                T.copy(A[pid_n * BLOCK_Q, k * BLOCK_Q], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                A_scale_1 = A_scale[pid_m, k] * A_scale[pid_n, k]
                for i, j in T.Parallel(BLOCK_Q, BLOCK_Q):
                    C_float[i, j] = T.cast(C_local[i, j], T.float32) * A_scale_1 + C_float[i, j]        
            
            C_scale_2 = T.alloc_reducer((1,), T.float32, op="max", replication="all")
            T.fill(C_scale_2, 0)
            if pid_m == pid_n:
                for i, j in T.Parallel(BLOCK_Q, BLOCK_Q):
                    if i == j:
                        C_diag[pid_m * BLOCK_Q + i] = C_float[i, j]
                        C_float[i, j] = 0
                    else:
                        C_scale_2[0] = T.max(C_scale_2[0], T.abs(C_float[i, j]))
            else:
                for i, j in T.Parallel(BLOCK_Q, BLOCK_Q):
                    C_scale_2[0] = T.max(C_scale_2[0], T.abs(C_float[i, j]))
            T.finalize_reducer(C_scale_2)
            
            A_scale_1 = DTYPE_MAX / C_scale_2[0]
            if (T.get_lane_idx() == 0 and T.get_warp_idx() == 0):
                C_scale[pid_m, pid_n] = 1 / A_scale_1
                C_scale[pid_n, pid_m] = 1 / A_scale_1

            for i, j in T.Parallel(BLOCK_Q, BLOCK_Q):
                C_float[i, j] *= A_scale_1
            
            C_shared = T.alloc_shared((BLOCK_Q, BLOCK_Q), dtype)
            D_shared = T.alloc_shared((BLOCK_Q, BLOCK_Q), dtype)
            if pid_m != pid_n:
                T.copy(C_float, C_shared)
                T.copy(C_shared, C[pid_m * BLOCK_Q, pid_n * BLOCK_Q])
                T.transpose(C_shared, D_shared)
                T.copy(D_shared, C[pid_n * BLOCK_Q, pid_m * BLOCK_Q])
            else:
                T.copy(C_float, C[pid_m * BLOCK_Q, pid_n * BLOCK_Q])

    return _aat_bq_


def _quad_bq(
    M: int = 4096,
    threads: int = 128, num_stages: int = 3,
    dtype: str = 'float8_e4m3fn',
):
    md = T.ceildiv(M, BLOCK_Q)
    total_blocks = md * (md + 1) // 2

    DTYPE_MAX = 448 if 'float8_e4m3' in dtype else 127 
    accum_dtype = 'float32' if 'float8_e4m3' in dtype else 'int32'
    
    @T.prim_func
    def _quad_bq_(
        A: T.Tensor((M, M), dtype),  # type: ignore
        A_scale: T.Tensor((md, md), 'float32'), # type: ignore
        A_diag: T.Tensor((M,), 'float32'), # type: ignore
        C: T.Tensor((M, M), dtype),  # type: ignore
        C_scale: T.Tensor((md, md), 'float32'), # type: ignore
        C_diag: T.Tensor((M,), 'float32'), # type: ignore
        C0: T.float32,
        C1: T.float32,
        C2: T.float32
    ):
        with T.Kernel(total_blocks, threads=threads) as (pid,):
            pid_m = T.alloc_var(T.int32)
            pid_m = T.cast(T.sqrt(8.0 * T.cast(pid, T.float32) + 1.0)*0.5 - 0.5, T.int32)
            pid_n = pid - (pid_m * (pid_m + 1) // 2)
            T.assume(pid_m >= 0)
            T.assume(pid_n >= 0)
            T.assume(pid_m < md)
            T.assume(pid_n < md)

            A_shared = T.alloc_shared((BLOCK_Q, BLOCK_Q), dtype)
            B_shared = T.alloc_shared((BLOCK_Q, BLOCK_Q), dtype)
            A_diag_shared = T.alloc_shared((BLOCK_Q,), T.float32)
            B_diag_shared = T.alloc_shared((BLOCK_Q,), T.float32)
            C_local = T.alloc_fragment((BLOCK_Q, BLOCK_Q), accum_dtype)
            C_float = T.alloc_fragment((BLOCK_Q, BLOCK_Q), T.float32)
            A_scale_1 = T.alloc_var(T.float32)

            T.clear(C_float)
            for k in T.Pipelined(md, num_stages=num_stages):
                T.clear(C_local)
                T.copy(A[pid_m * BLOCK_Q, k * BLOCK_Q], A_shared)
                T.copy(A[pid_n * BLOCK_Q, k * BLOCK_Q], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                A_scale_1 = A_scale[pid_m, k] * A_scale[pid_n, k]
                for i, j in T.Parallel(BLOCK_Q, BLOCK_Q):
                    C_float[i, j] = T.cast(C_local[i, j], T.float32) * A_scale_1 + C_float[i, j] 

            T.copy(A_diag[pid_m * BLOCK_Q], A_diag_shared)
            T.copy(A_diag[pid_n * BLOCK_Q], B_diag_shared)
            A_shared_2 = T.alloc_fragment((BLOCK_Q, BLOCK_Q), dtype)
            A_scale_1 = A_scale[pid_m, pid_n]
            T.copy(A[pid_m * BLOCK_Q, pid_n * BLOCK_Q], A_shared_2)
            is_diag = T.alloc_var(T.bool)
            is_diag = pid_m == pid_n

            C_scale_2 = T.alloc_reducer((1,), T.float32, op="max", replication="all")
            T.fill(C_scale_2, 0)
            ai = T.alloc_var(T.float32)
            for i, j in T.Parallel(BLOCK_Q, BLOCK_Q):
                ai = A_diag_shared[i]
                C_float[i, j] += (ai + B_diag_shared[j]) * (A_scale_1 * A_shared_2[i, j])
                C_float[i, j] *= C2
                C_float[i, j] += (A_scale_1 * A_shared_2[i, j]) * C1
                if is_diag and i == j:
                    C_diag[pid_m * BLOCK_Q + i] = C_float[i, j] + (C0 + ai * (C1 + ai * C2))
                    C_float[i, j] = 0
                else:
                    C_scale_2[0] = T.max(C_scale_2[0], T.abs(C_float[i, j]))
            T.finalize_reducer(C_scale_2)

            A_scale_1 = DTYPE_MAX / C_scale_2[0]
            if (T.get_lane_idx() == 0 and T.get_warp_idx() == 0):
                C_scale[pid_m, pid_n] = 1 / A_scale_1
                C_scale[pid_n, pid_m] = 1 / A_scale_1

            for i, j in T.Parallel(BLOCK_Q, BLOCK_Q):
                C_float[i, j] *= A_scale_1
            
            C_shared = T.alloc_shared((BLOCK_Q, BLOCK_Q), dtype)
            D_shared = T.alloc_shared((BLOCK_Q, BLOCK_Q), dtype)
            if pid_m != pid_n:
                T.copy(C_float, C_shared)
                T.copy(C_shared, C[pid_m * BLOCK_Q, pid_n * BLOCK_Q])
                T.transpose(C_shared, D_shared)
                T.copy(D_shared, C[pid_n * BLOCK_Q, pid_m * BLOCK_Q])
            else:
                T.copy(C_float, C[pid_m * BLOCK_Q, pid_n * BLOCK_Q])
    return _quad_bq_

def _typeii_typei_bq(
    M: int = 4096, N: int= 6144,
    threads: int = 128, num_stages: int = 3,
    dtype: str = 'float8_e4m3fn'
):
    md = T.ceildiv(M, BLOCK_Q)
    nd = T.ceildiv(N, BLOCK_Q)

    DTYPE_MAX = 448 if 'float8_e4m3' in dtype else 127 
    accum_dtype = 'float32' if 'float8_e4m3' in dtype else 'int32'
    @T.prim_func
    def _typeii_typei_bq_(
        A:       T.Tensor((M, M),   dtype),     # type: ignore
        A_scale: T.Tensor((md, md), T.float32), # type: ignore
        A_diag:  T.Tensor((M,),     T.float32), # type: ignore
        B:       T.Tensor((M, N),   dtype),     # type: ignore
        B_scale: T.Tensor((md, nd), T.float32), # type: ignore
        # B_diag: T.Tensor((M,), T.float32),
        C:       T.Tensor((M, N),   dtype),     # type: ignore
        C_scale: T.Tensor((md, nd), T.float32), # type: ignore
    ):
        with T.Kernel(nd, md, threads=threads) as (pid_n, pid_m):
            A_shared = T.alloc_shared((BLOCK_Q, BLOCK_Q), dtype)
            B_shared = T.alloc_shared((BLOCK_Q, BLOCK_Q), dtype)
            row_shared = T.alloc_shared((BLOCK_Q,), T.float32)
            C_local = T.alloc_fragment((BLOCK_Q, BLOCK_Q), accum_dtype)
            C_float = T.alloc_fragment((BLOCK_Q, BLOCK_Q), T.float32)
            A_scale_1 = T.alloc_var(T.float32)

            T.clear(C_float)
            for k in T.Pipelined(md, num_stages=num_stages):
                T.clear(C_local)
                T.copy(A[pid_m * BLOCK_Q, k * BLOCK_Q], A_shared)
                T.copy(B[k * BLOCK_Q, pid_n * BLOCK_Q], B_shared)
                T.gemm(A_shared, B_shared, C_local)
                A_scale_1 = A_scale[pid_m, k] * B_scale[k, pid_n]
                for i, j in T.Parallel(BLOCK_Q, BLOCK_Q):
                    C_float[i, j] = T.cast(C_local[i, j], T.float32) * A_scale_1 + C_float[i, j]

            B8 = T.alloc_fragment((BLOCK_Q, BLOCK_Q), dtype)
            T.copy(A_diag[pid_m * BLOCK_Q], row_shared)
            T.copy(B[pid_m * BLOCK_Q, pid_n * BLOCK_Q], B8)
            B_var = T.alloc_var(T.float32)
            B_var = B_scale[pid_m, pid_n]
            # A_var_beta = A_var * BETA
            Cmax_reducer = T.alloc_reducer((1,), T.float32, op='max', replication="all")
            T.fill(Cmax_reducer, 0)
            for (i, j) in T.Parallel(BLOCK_Q, BLOCK_Q):
                C_float[i, j] += (T.cast(B8[i,j], T.float32) * row_shared[i]) * B_var
                val = T.abs(C_float[i, j])
                Cmax_reducer[0] = T.max(val, Cmax_reducer[0])
            T.finalize_reducer(Cmax_reducer)

            A_scale_1 = DTYPE_MAX / Cmax_reducer[0]
            if (T.get_lane_idx() == 0 and T.get_warp_idx() == 0):
                C_scale[pid_m, pid_n] = 1 / A_scale_1
            for i, j in T.Parallel(BLOCK_Q, BLOCK_Q):
                C_float[i, j] = C_float[i, j] * A_scale_1
            T.copy(C_float, C[pid_m * BLOCK_Q, pid_n * BLOCK_Q])
            
    return _typeii_typei_bq_


def _typeii_typei_final_bq(
    M: int = 4096, N: int= 6144,
    threads: int = 128, num_stages: int = 3,
    dtype: str = 'float8_e4m3fn',
    dtype2: str = 'float32',
):
    md = T.ceildiv(M, BLOCK_Q)
    nd = T.ceildiv(N, BLOCK_Q)
    accum_dtype = 'float32' if 'float8_e4m3' in dtype else 'int32'

    @T.prim_func
    def _typeii_typei_final_bq_(
        A:       T.Tensor((M, M), dtype), # type: ignore
        A_scale: T.Tensor((md, md), T.float32), # type: ignore
        A_diag:  T.Tensor((M,), T.float32), # type: ignore
        B:       T.Tensor((M, N), dtype), # type: ignore
        B_scale: T.Tensor((md, nd), T.float32), # type: ignore
        # B_diag: T.TensNor((M,), T.float32),
        C:       T.Tensor((M, N), dtype2), # type: ignore
    ):
        with T.Kernel(nd, md, threads=threads) as (pid_n, pid_m):
            A_shared = T.alloc_shared((BLOCK_Q, BLOCK_Q), dtype)
            B_shared = T.alloc_shared((BLOCK_Q, BLOCK_Q), dtype)
            row_shared = T.alloc_shared((BLOCK_Q,), T.float32)
            C_local = T.alloc_fragment((BLOCK_Q, BLOCK_Q), accum_dtype)
            C_float = T.alloc_fragment((BLOCK_Q, BLOCK_Q), T.float32)
            A_scale_1 = T.alloc_var(T.float32)

            T.clear(C_float)
            for k in T.Pipelined(md, num_stages=num_stages):
                T.clear(C_local)
                T.copy(A[pid_m * BLOCK_Q, k * BLOCK_Q], A_shared)
                T.copy(B[k * BLOCK_Q, pid_n * BLOCK_Q], B_shared)
                T.gemm(A_shared, B_shared, C_local)
                A_scale_1 = A_scale[pid_m, k] * B_scale[k, pid_n]
                for i, j in T.Parallel(BLOCK_Q, BLOCK_Q):
                    C_float[i, j] = T.cast(C_local[i, j], T.float32) * A_scale_1 + C_float[i, j]

            B8 = T.alloc_fragment((BLOCK_Q, BLOCK_Q), dtype)
            T.copy(A_diag[pid_m * BLOCK_Q], row_shared)
            T.copy(B[pid_m * BLOCK_Q, pid_n * BLOCK_Q], B8)
            B_var = T.alloc_var(T.float32)
            B_var = B_scale[pid_m, pid_n]

            for (i, j) in T.Parallel(BLOCK_Q, BLOCK_Q):
                C_float[i, j] += (T.cast(B8[i,j], T.float32) * row_shared[i]) * B_var

            T.copy(C_float, C[pid_m * BLOCK_Q, pid_n * BLOCK_Q])

    return _typeii_typei_final_bq_


def _ab_symm_bq(
    M: int = 4096,
    threads: int = 128, num_stages: int = 3,
    dtype: str = 'float8_e4m3fn',
):
    md = T.ceildiv(M, BLOCK_Q)
    total_blocks = md * (md + 1) // 2

    DTYPE_MAX = 448 if 'float8_e4m3' in dtype else 127 
    accum_dtype = 'float32' if 'float8_e4m3' in dtype else 'int32'
    
    @T.prim_func
    def _ab_symm_bq_(
        A: T.Tensor((M, M), dtype),  # type: ignore
        A_scale: T.Tensor((md, md), 'float32'), # type: ignore
        A_diag: T.Tensor((M,), 'float32'), # type: ignore
        B: T.Tensor((M, M), dtype),  # type: ignore
        B_scale: T.Tensor((md, md), 'float32'), # type: ignore
        B_diag: T.Tensor((M,), 'float32'), # type: ignore
        C: T.Tensor((M, M), dtype),  # type: ignore
        C_scale: T.Tensor((md, md), 'float32'), # type: ignore
        C_diag: T.Tensor((M,), 'float32'), # type: ignore
    ):
        with T.Kernel(total_blocks, threads=threads) as (pid,):
            pid_m = T.alloc_var(T.int32)
            pid_m = T.cast(T.sqrt(8.0 * T.cast(pid, T.float32) + 1.0)*0.5 - 0.5, T.int32)
            pid_n = pid - (pid_m * (pid_m + 1) // 2)
            T.assume(pid_m >= 0)
            T.assume(pid_n >= 0)
            T.assume(pid_m < md)
            T.assume(pid_n < md)

            A_shared = T.alloc_shared((BLOCK_Q, BLOCK_Q), dtype)
            B_shared = T.alloc_shared((BLOCK_Q, BLOCK_Q), dtype)
            A_diag_shared = T.alloc_shared((BLOCK_Q,), T.float32)
            B_diag_shared = T.alloc_shared((BLOCK_Q,), T.float32)
            C_local = T.alloc_fragment((BLOCK_Q, BLOCK_Q), accum_dtype)
            C_float = T.alloc_fragment((BLOCK_Q, BLOCK_Q), T.float32)
            A_scale_1 = T.alloc_var(T.float32)
            B_scale_1 = T.alloc_var(T.float32)

            T.clear(C_float)
            for k in T.Pipelined(md, num_stages=num_stages):
                T.clear(C_local)
                T.copy(A[pid_m * BLOCK_Q, k * BLOCK_Q], A_shared)
                T.copy(B[pid_n * BLOCK_Q, k * BLOCK_Q], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                A_scale_1 = A_scale[pid_m, k] * B_scale[pid_n, k]
                for i, j in T.Parallel(BLOCK_Q, BLOCK_Q):
                    C_float[i, j] = T.cast(C_local[i, j], T.float32) * A_scale_1 + C_float[i, j] 

            T.copy(A_diag[pid_m * BLOCK_Q], A_diag_shared)
            T.copy(B_diag[pid_n * BLOCK_Q], B_diag_shared)
            A_shared_2 = T.alloc_fragment((BLOCK_Q, BLOCK_Q), dtype)
            B_shared_2 = T.alloc_fragment((BLOCK_Q, BLOCK_Q), dtype)
            A_scale_1 = A_scale[pid_m, pid_n]
            B_scale_1 = B_scale[pid_m, pid_n]
            T.copy(A[pid_m * BLOCK_Q, pid_n * BLOCK_Q], A_shared_2)
            T.copy(B[pid_m * BLOCK_Q, pid_n * BLOCK_Q], B_shared_2)
            
            is_diag = T.alloc_var(T.bool)
            is_diag = pid_m == pid_n

            C_scale_2 = T.alloc_reducer((1,), T.float32, op="max", replication="all")
            T.fill(C_scale_2, 0)
            for i, j in T.Parallel(BLOCK_Q, BLOCK_Q):
                C_float[i, j] += (
                    B_diag_shared[j] * A_scale_1 * A_shared_2[i, j] 
                    + A_diag_shared[i] * B_scale_1 * B_shared_2[i, j]
                )
                if is_diag and i == j:
                    C_diag[pid_m * BLOCK_Q + i] = C_float[i, j] + A_diag_shared[i] * B_diag_shared[j]
                    C_float[i, j] = 0
                else:
                    C_scale_2[0] = T.max(C_scale_2[0], T.abs(C_float[i, j]))
            T.finalize_reducer(C_scale_2)

            A_scale_1 = DTYPE_MAX / C_scale_2[0]
            if (T.get_lane_idx() == 0 and T.get_warp_idx() == 0):
                C_scale[pid_m, pid_n] = 1 / A_scale_1
                C_scale[pid_n, pid_m] = 1 / A_scale_1

            for i, j in T.Parallel(BLOCK_Q, BLOCK_Q):
                C_float[i, j] *= A_scale_1
            
            C_shared = T.alloc_shared((BLOCK_Q, BLOCK_Q), dtype)
            D_shared = T.alloc_shared((BLOCK_Q, BLOCK_Q), dtype)
            if pid_m != pid_n:
                T.copy(C_float, C_shared)
                T.copy(C_shared, C[pid_m * BLOCK_Q, pid_n * BLOCK_Q])
                T.transpose(C_shared, D_shared)
                T.copy(D_shared, C[pid_n * BLOCK_Q, pid_m * BLOCK_Q])
            else:
                T.copy(C_float, C[pid_m * BLOCK_Q, pid_n * BLOCK_Q])
    return _ab_symm_bq_



__all__ = [
    # global int8
    '_sumsq_maxabs', '_scale_int8', '_aat_int8_max', '_int32_compl_symm_int8', '_typeii_int8_sq',
    '_float32_compl_symm_int8_quad', '_typeii_int8_ab', '_float32_ab_to_int8', '_typeii_typei_int8',
    '_float32_to_int8', 
    # fp16/bf16/fp32
    '_to_prec', '_ab_prec', '_aat_prec', '_quad_prec', '_ab_symm_prec', 
    # block fp8/int8
    '_to_bq', '_aat_bq','_quad_bq', '_typeii_typei_bq', '_typeii_typei_final_bq', '_ab_symm_bq',
    'BLOCK_Q'
]
