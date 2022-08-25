#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Layout.h>
#include <ATen/Parallel.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/NativeFunctions.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/SparseTensorUtils.h>

#include "cusparse.h"

#include <pybind11/pybind11.h>

// #include <THC/THCGeneral.hpp>

#include <torch/extension.h>

namespace py = pybind11;

using namespace at::sparse;

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
    }                                                                          \
}

#define CHECK_ERROR(str) \
    {cudaDeviceSynchronize(); cudaError_t err; err = cudaGetLastError(); if(err!=0) {printf("ERROR %s:  %d %s\n", str, err, cudaGetErrorString(err)); fflush(stdout);}}

__device__ int binary_searchf(double *arr, double val, int imin, int imax) {
    
    int ans = 0;
    while (imax >= imin) {
        int imid = (imin + imax) / 2;

        if (arr[imid] < val) {
            imin = imid + 1;
        } else {
            ans = imid;
            imax = imid - 1;
        }
    }

    return ans;
}

__device__ int binary_searchfpf(double *arr, double val, int imin, int imax) {

    int ans = 0;
    while (imax >= imin) {
        int imid = (imin + imax) / 2;
        printf("imid: %d val: %f arr[imid]: %f\n", imid, val, arr[imid]);

        if (arr[imid] < val) {
            imin = imid + 1;
        } else {
            ans = imid;
            imax = imid - 1;
        }
    }

    return ans;
}

__device__ long binary_searchl(long *arr, long val, long imin, long imax) {
    
    long ans = -1;
    while (imax >= imin) {
        long imid = (imin + imax) / 2;

        if (arr[imid] <= val) {
            imin = imid + 1;
        } else if (arr[imid] > val) {
            ans = imid;
            imax = imid - 1;
        }
    }

    return ans;
}

// Binary search that returns exact location, not insertion point (for COO row selection)
__device__ long binary_search_rowselect(long *arr, long val, long imin, long imax) {
    while (imax >= imin) {
        long imid = (imin + imax) / 2;

        if (arr[imid] < val) {
            imin = imid + 1;
        } else if (arr[imid] > val) {
            imax = imid - 1;
        } else {
            return imid;
        }
    }

    return imin + 1;
}

at::Tensor expand_values_if_needed(const at::Tensor& values) {
    // expand
    if (values.dim() == 0) {
        // Mimic Numpy behavior here and treat it as a 1D tensor
        return values.expand({1});
    } else {
        return values;
    }
}

at::Tensor sparse_coo_tensor_gpu(const at::Tensor& indices, 
                                    const at::Tensor& values_, 
                                    at::ArrayRef<int64_t> size) {

    at::Tensor values = expand_values_if_needed(values_); 

    int64_t sparse_dim = indices.size(0);
    int64_t dense_dim = values.dim() - 1;

    return at::_sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim, dense_dim, size, indices, values, values.options().layout(at::kSparse));
}

template<typename T>
void printCusparseDnMat(int64_t rows, int64_t cols, int64_t ld, T *values_dev) {
  T* values_host = new T[rows*cols];
  cudaMemcpy(values_host, values_dev, rows*cols*sizeof(T), cudaMemcpyDeviceToHost);
  for (int64_t row = 0; row < rows; row++) {
    for (int64_t col = 0; col < cols; col++) {
      // Cusparse dense matrices are stored in column-major order
      std::cout << values_host[col*rows+row] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "  values: ";
  for (int64_t i = 0; i < rows*cols; i++) {
    std::cout << values_host[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "  shape: " << rows << ", " << cols << std::endl;
  delete [] values_host;
}

template<typename T>
void printCusparseSpMat(int32_t rows, int32_t cols, int32_t nnz, int32_t *row_indices_dev,
                            int32_t *col_indices_dev, T *values_dev) {
  T* values_host = new T[nnz];
  int32_t* row_indices_host = new int32_t[nnz];
  int32_t* col_indices_host = new int32_t[nnz];
  cudaMemcpy(values_host, values_dev, nnz*sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(row_indices_host, row_indices_dev, nnz*sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(col_indices_host, col_indices_dev, nnz*sizeof(int32_t), cudaMemcpyDeviceToHost);

  for (int64_t i = 0; i < nnz; i++) {
    std::cout << "(" << row_indices_host[i]
      << ", " << col_indices_host[i]
      << "): " << values_host[i] << std::endl;
  }
  std::cout << "  values: ";
  for (int64_t i = 0; i < nnz; i++) {
    std::cout << values_host[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "  row_indices: ";
  for (int64_t i = 0; i < nnz; i++) {
    std::cout << row_indices_host[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "  col_indices: ";
  for (int64_t i = 0; i < nnz; i++) {
    std::cout << col_indices_host[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "  shape: " << rows << ", " << cols << std::endl;
  delete [] values_host;
  delete [] row_indices_host;
  delete [] col_indices_host;
}

// at::Tensor spmm_gpu(const at::Tensor& A_rowindices, 
void spmm_gpu(const at::Tensor& A_rowindices, 
                        const at::Tensor& A_colindices,
                        const at::Tensor& A_values, 
                        int32_t n,
                        int32_t m,
                        at::Tensor& B,
                        at::Tensor& C) {

    // cusparseHandle_t handle;
    // CHECK_CUSPARSE(cusparseCreate(&handle));
    // auto state = at::globalContext().lazyInitCUDA();
    // auto handle = THCState_getCurrentSparseHandle(state);
    auto handle = at::cuda::getCurrentCUDASparseHandle();

    // Impl1 -- coo2csr + csrmm2
    int nnz = A_values.size(0);

    clock_t start, stop;
    
    int32_t *d_a_csrrows;
    
    // int devid_old = 0;
    // cudaGetDevice(&devid_old);
    // cudaSetDevice(devid);

    cudaMalloc(&d_a_csrrows, (n + 1) * sizeof(int32_t));
    CHECK_CUSPARSE(cusparseXcoo2csr(handle, 
                                        A_rowindices.data<int>(), 
                                        nnz, 
                                        n, 
                                        d_a_csrrows, 
                                        CUSPARSE_INDEX_BASE_ZERO));

    int32_t b_row = B.size(0);
    int32_t b_col = B.size(1);
    int32_t c_row = C.size(0);
    int32_t c_col = C.size(1);

    float alpha = 1;
    float beta = 1;
    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA,
					  n, 		// rows
					  m, 	        // cols
					  nnz, 		// nnz
					  d_a_csrrows, 	// csrRowOffsets
					  A_colindices.data<int>(), // csrColInd
					  A_values.data<float>(),   // csrValues
					  CUSPARSE_INDEX_32I, 	    // csrRowOffsetsType
					  CUSPARSE_INDEX_32I, 	    // csrColIndType
					  CUSPARSE_INDEX_BASE_ZERO, // idxBase,
					  CUDA_R_32F)); 	    // valueType

    cusparseDnMatDescr_t matB;
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, 
                                            b_col, // rows
                                            b_row, // cols
                                            b_col, // ld
                                            B.data<float>(), // values
                                            CUDA_R_32F,      // valueType
                                            CUSPARSE_ORDER_COL)); // order
        
    // Row-major to column-major
    C.t_();
    C.set_data(C.contiguous());
    C.set_data(C.view({c_row, c_col}));

    cusparseDnMatDescr_t matC;
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, 
                                            n, // rows
                                            b_col, // cols
                                            n, // ld
                                            C.data<float>(), // values
                                            CUDA_R_32F,      // valueType
                                            CUSPARSE_ORDER_COL)); // order
	
    size_t bufferSize;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, // handle,
                                                CUSPARSE_OPERATION_NON_TRANSPOSE,   // opA
                                                CUSPARSE_OPERATION_TRANSPOSE,   // opB
                                                &alpha,                             // alpha
                                                matA,                               // matA
                                                matB,                               // matB
                                                &beta,                              // beta
                                                matC,                               // matC
                                                CUDA_R_32F,                         // computeType
                                                CUSPARSE_CSRMM_ALG1,                // alg
                                                &bufferSize));                      // bufferSize


    void* d_buffer = NULL;
    CHECK_ERROR(cudaMalloc(&d_buffer, bufferSize));

    CHECK_CUSPARSE(cusparseSpMM(handle, // handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,   // opA
                                    CUSPARSE_OPERATION_TRANSPOSE,   // opB
                                    &alpha,                             // alpha
                                    matA,                               // matA
                                    matB,                               // matB
                                    &beta,                              // beta
                                    matC,                               // matC
                                    CUDA_R_32F,                         // computeType
                                    CUSPARSE_CSRMM_ALG1,                // alg
                                    d_buffer));                         // buffer


    cudaFree(d_a_csrrows);
    cudaFree(d_buffer);

    // Column-major to row-major
    C.set_data(C.view({c_col, c_row}));
    C.t_();
}

// __global__ void DownSample(long *q_values, long *q_rows, int *overflow, int nnz) {
//     int     id = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
// 
//     for (int i = id; i < nnz; i += stride) {
//         long row = q_rows[i];
//         if (q_values[i] == 1 && ((int) atomicSub((unsigned int *)&overflow[row], 1)) > 0) {
//             q_values[i] = 0;
//         }
//     }
// }

// void downsample_gpu(const at::Tensor& q_values, 
//                         const at::Tensor& q_rows,
//                         const at::Tensor& overflow,
//                         int nnz) {
// 
// 
//     int BLOCK_SIZE = 256;
//     int BLOCK_COUNT = std::ceil(nnz / ((float) BLOCK_SIZE));
//     BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);
// 
//     DownSample<<<BLOCK_COUNT, BLOCK_SIZE>>>(q_values.data<long>(), q_rows.data<long>(), 
//                                                                 overflow.data<int>(), nnz);
//     CHECK_ERROR("downsampling error")
// }

__global__ void DownSample(long *h_counts, long *h_rows, long *ps_h_rows, long *hev_indices, int *overflow, 
                                int nnz) {

    long     id = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    for (long i = id; i < nnz; i += stride) {
        if (i - ps_h_rows[h_rows[i]] < overflow[h_rows[i]]) {
            h_counts[hev_indices[i]] = 0;
        }
    }
}

void downsample_gpu(const at::Tensor& h_counts, 
                        const at::Tensor& h_rows,
                        const at::Tensor& ps_h_rows,
                        const at::Tensor& hev_indices,
                        const at::Tensor& overflow,
                        int nnz) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(nnz / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    DownSample<<<BLOCK_COUNT, BLOCK_SIZE>>>(h_counts.data<long>(), 
                                                h_rows.data<long>(), 
                                                ps_h_rows.data<long>(), 
                                                hev_indices.data<long>(), 
                                                overflow.data<int>(), 
                                                nnz);
    CHECK_ERROR("downsampling error")
}

__global__ void ComputeDarts(float *dartx_values, float *darty_values, long *neighbor_sizes, 
                                long *psum_neighbor_sizes, float *pmax, int n_darts, int mb_count) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int dart_count = n_darts * mb_count;
    for (int i = id; i < dart_count; i += stride) {
        long row = i / n_darts;
        dartx_values[i] *= neighbor_sizes[row];
        dartx_values[i] += psum_neighbor_sizes[row];
        darty_values[i] *= pmax[row];
    }
}

void compute_darts_gpu(const at::Tensor& dartx_values, 
                        const at::Tensor& darty_values,
                        const at::Tensor& neighbor_sizes,
                        const at::Tensor& psum_neighbor_sizes,
                        const at::Tensor& pmax,
                        int n_darts,
                        int mb_count) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil((n_darts * mb_count) / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    ComputeDarts<<<BLOCK_COUNT, BLOCK_SIZE>>>(dartx_values.data<float>(), 
                                                darty_values.data<float>(), 
                                                neighbor_sizes.data<long>(), 
                                                psum_neighbor_sizes.data<long>(), 
                                                pmax.data<float>(), 
                                                n_darts,
                                                mb_count);
    CHECK_ERROR("dart computation error")
}

__global__ void ComputeDarts1D(double *dart_values, int n_darts, int mb_count) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int dart_count = n_darts * mb_count;
    for (int i = id; i < dart_count; i += stride) {
        int row = i / n_darts;
        dart_values[i] += (double)row;
    }
}

void compute_darts1d_gpu(const at::Tensor& dart_values, int n_darts, int mb_count) {

    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil((n_darts * mb_count) / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    ComputeDarts1D<<<BLOCK_COUNT, BLOCK_SIZE>>>(dart_values.data<double>(), 
                                                n_darts,
                                                mb_count);
    CHECK_ERROR("dart1d computation error")
}

__global__ void ComputeDartsSelect(double *dart_select, double *dart_hits_inv_sum, double *ps_dart_hits_inv_sum, 
                                long *ps_overflow, long mb_count, long total_overflow) {

    long      id = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    for (long i = id; i < total_overflow; i += stride) {
        long row = binary_searchl(ps_overflow, i, 0, mb_count);
        dart_select[i] *= dart_hits_inv_sum[row];
        dart_select[i] += ps_dart_hits_inv_sum[row];
    }
}

void compute_darts_select_gpu(const at::Tensor& dart_select, 
                                const at::Tensor& dart_hits_inv_sum,
                                const at::Tensor& ps_dart_hits_inv_sum,
                                const at::Tensor& ps_overflow,
                                long mb_count,
                                long total_overflow) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(total_overflow / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    ComputeDartsSelect<<<BLOCK_COUNT, BLOCK_SIZE>>>(dart_select.data<double>(), 
                                                        dart_hits_inv_sum.data<double>(), 
                                                        ps_dart_hits_inv_sum.data<double>(), 
                                                        ps_overflow.data<long>(), 
                                                        mb_count,
                                                        total_overflow);
    CHECK_ERROR("selection dart computation error")
}

__global__ void ThrowDarts(float *dartx_values, float *darty_values, float *p_values, 
                                long *h_values, int n_darts, int mb_count) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int dart_count = n_darts * mb_count;
    for (int i = id; i < dart_count; i += stride) {
        long dartx_val = (long) dartx_values[i];
        if (darty_values[i] < p_values[dartx_val]) {
            atomicAdd((unsigned long long *)&h_values[dartx_val], 1L);
        }
    }
}

void throw_darts_gpu(const at::Tensor& dartx_values, 
                        const at::Tensor& darty_values,
                        const at::Tensor& p_values,
                        const at::Tensor& h_values,
                        int n_darts,
                        int mb_count) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil((n_darts * mb_count) / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    ThrowDarts<<<BLOCK_COUNT, BLOCK_SIZE>>>(dartx_values.data<float>(), 
                                                darty_values.data<float>(), 
                                                p_values.data<float>(), 
                                                h_values.data<long>(), 
                                                n_darts,
                                                mb_count);
    CHECK_ERROR("dart throwing error")
}

__global__ void ThrowDarts1D(double *dart_values, double *ps_p_values, int *h_values, 
                                int dart_count, int nnz) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < dart_count; i += stride) {
        int vtx = binary_searchf(ps_p_values, dart_values[i], 0, nnz - 1);
        if (vtx < 0 || vtx >= nnz) {
            printf("error i: %d vtx: %d nnz: %d\n", i, vtx, nnz);
        } 
        atomicAdd(&h_values[vtx], 1);
    }
}

void throw_darts1d_gpu(const at::Tensor& dart_values, 
                            const at::Tensor& ps_p_values,
                            const at::Tensor& h_values,
                            int dart_count,
                            int nnz) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil((dart_count) / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    ThrowDarts1D<<<BLOCK_COUNT, BLOCK_SIZE>>>(dart_values.data<double>(), 
                                                ps_p_values.data<double>(), 
                                                h_values.data<int>(), 
                                                dart_count,
                                                nnz);
    CHECK_ERROR("dart throwing error")
}

__global__ void ThrowDartsSelect(double *dart_select, double *ps_dart_hits_inv, int *dart_hits_count, 
                                    int total_overflow, int nnz) {

    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < total_overflow; i += stride) {
        int vtx = binary_searchf(ps_dart_hits_inv, dart_select[i], 0, nnz);
        atomicAnd(&dart_hits_count[vtx], 0);
    }
}

void throw_darts_select_gpu(const at::Tensor& dart_select, 
                                const at::Tensor& ps_dart_hits_inv,
                                const at::Tensor& dart_hits_count,
                                int total_overflow,
                                int nnz) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(total_overflow / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    ThrowDartsSelect<<<BLOCK_COUNT, BLOCK_SIZE>>>(dart_select.data<double>(), 
                                                    ps_dart_hits_inv.data<double>(), 
                                                    dart_hits_count.data<int>(), 
                                                    total_overflow,
                                                    nnz);
    CHECK_ERROR("selection dart throwing error")
}

__global__ void Normalize(double *output, double *input, long *index, int len) { 
    long     id = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    for (int i = id; i < len; i += stride) {
        output[i] /= input[index[i]];
    }
}

void normalize_gpu(const at::Tensor& output, const at::Tensor& input, const at::Tensor& index, int len) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(len / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    Normalize<<<BLOCK_COUNT, BLOCK_SIZE>>>(output.data<double>(), input.data<double>(), index.data<long>(), len);
    CHECK_ERROR("normalize error")
}

__global__ void ShiftRowSelect(long *row_shift, long *row_select_rows, int rank, int size,
                                    int replication, int nnz, int batch_size, int node_count, int mb_count) { 

    long     id = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    int rank_c = rank / replication;
    long proc_row_chunk = rank_c * ((batch_size * mb_count) / (size / replication));
    
    for (int i = id; i < nnz; i += stride) {
        long mb_row = (row_select_rows[i] + proc_row_chunk) / batch_size;
        row_shift[i] += mb_row * node_count;
    }
}

void shift_rowselect_gpu(const at::Tensor& row_shift, const at::Tensor& row_select_rows,
                            int nnz, int rank, int size, int replication, int batch_size, int node_count, 
                            int mb_count) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(nnz / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    ShiftRowSelect<<<BLOCK_COUNT, BLOCK_SIZE>>>(row_shift.data<long>(), 
                                                    row_select_rows.data<long>(), 
                                                    rank,
                                                    size,
                                                    replication,
                                                    nnz,
                                                    batch_size,
                                                    node_count,
                                                    mb_count);
    CHECK_ERROR("shift row select error")
}

__global__ void ShiftColSelect(long *col_shift, int nnz, int batch_size, int node_count) { 
    long     id = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;

    for (int i = id; i < nnz; i += stride) {
        long mb_row = i / batch_size;
        col_shift[i] += mb_row * node_count;
    }
}

void shift_colselect_gpu(const at::Tensor& col_shift, int nnz, int batch_size, int node_count) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(nnz / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    ShiftColSelect<<<BLOCK_COUNT, BLOCK_SIZE>>>(col_shift.data<long>(), nnz, batch_size, node_count);
    CHECK_ERROR("shift col select error")
}

__global__ void ScatterAddD(double *src, long *indices, double *values, int num_vals) { 
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < num_vals; i += stride) {
        atomicAdd(&src[indices[i]], values[i]);
    } 
}

void scatterd_add_gpu(const at::Tensor& src, const at::Tensor& indices, const at::Tensor& values, int num_vals) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(num_vals / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    ScatterAddD<<<BLOCK_COUNT, BLOCK_SIZE>>>(src.data<double>(), 
                                                indices.data<long>(), 
                                                values.data<double>(),
                                                num_vals);
    CHECK_ERROR("scatter add doubles error")
}

__global__ void ScatterAddI(int *src, long *indices, int *values, int num_vals) { 
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < num_vals; i += stride) {
        atomicAdd(&src[indices[i]], values[i]);
    } 
}

void scatteri_add_gpu(const at::Tensor& src, const at::Tensor& indices, const at::Tensor& values, int num_vals) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(num_vals / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    ScatterAddI<<<BLOCK_COUNT, BLOCK_SIZE>>>(src.data<int>(), 
                                                indices.data<long>(), 
                                                values.data<int>(),
                                                num_vals);
    CHECK_ERROR("scatter add ints error")
}

__global__ void RowSelectCoo(long *nnz_cols, long *row_ids, bool *mask, int nnz_col_count, int row_count) { 
    int     id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < row_count; i += stride) {
        long idx = binary_search_rowselect(nnz_cols, row_ids[i], 0L, nnz_col_count);
        if (nnz_cols[idx] == row_ids[i]) {
            mask[i] = true;
        }
    } 
}

void rowselect_coo_gpu(const at::Tensor& nnz_cols, const at::Tensor& rows, const at::Tensor& mask, 
                            int nnz_col_count, int row_count) {


    int BLOCK_SIZE = 256;
    int BLOCK_COUNT = std::ceil(row_count / ((float) BLOCK_SIZE));
    BLOCK_COUNT = std::min(BLOCK_COUNT, 65535);

    if (nnz_col_count == 0) {
        return;
    }

    RowSelectCoo<<<BLOCK_COUNT, BLOCK_SIZE>>>(nnz_cols.data<long>(), 
                                                rows.data<long>(), 
                                                mask.data<bool>(), 
                                                nnz_col_count,
                                                row_count);
    CHECK_ERROR("rowselect coo error")
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_coo_tensor_gpu", &sparse_coo_tensor_gpu, "Sparse Tensor GPU-only constructor");
    m.def("spmm_gpu", &spmm_gpu, "SpMM wrapper for cusparse");
    m.def("downsample_gpu", &downsample_gpu, "Downsampling for LADIES sampling algorithm");
    m.def("compute_darts_gpu", &compute_darts_gpu, "Compute dart values for LADIES sampling algorithm");
    m.def("throw_darts_gpu", &throw_darts_gpu, "Throw darts in LADIES sampling algorithm");
    m.def("compute_darts_select_gpu", &compute_darts_select_gpu, "Compute dart values for LADIES alg selection");
    m.def("throw_darts_select_gpu", &throw_darts_select_gpu, "Throw darts for LADIES alg selection");
    m.def("compute_darts1d_gpu", &compute_darts1d_gpu, "Compute 1D dart values for LADIES sampling algorithm");
    m.def("throw_darts1d_gpu", &throw_darts1d_gpu, "Throw 1D darts in LADIES sampling algorithm");
    m.def("normalize_gpu", &normalize_gpu, "Normalize values in an array based on a second array");
    m.def("shift_rowselect_gpu", &shift_rowselect_gpu, "Shift row selection output matrix col values");
    m.def("shift_colselect_gpu", &shift_colselect_gpu, "Shift col selection matrix row values");
    m.def("scatterd_add_gpu", &scatterd_add_gpu, "Implementation of scatter_add_ for doubles");
    m.def("scatteri_add_gpu", &scatteri_add_gpu, "Implementation of scatter_add_ for ints");
    m.def("rowselect_coo_gpu", &rowselect_coo_gpu, "Row selection for sparsity-aware spgemm");
}
