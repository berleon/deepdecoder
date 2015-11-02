// Kernel definition

#define MASK_LEN 29
#define WARP_SIZE 32


// TODO: include deepdecoder headers for mask enum
enum MASK {
    INNER_BLACK_SEMICIRCLE,
    CELL_0_BLACK = 1,
    CELL_1_BLACK,
    CELL_2_BLACK,
    CELL_3_BLACK,
    CELL_4_BLACK,
    CELL_5_BLACK,
    CELL_6_BLACK,
    CELL_7_BLACK,
    CELL_8_BLACK,
    CELL_9_BLACK,
    CELL_10_BLACK,
    CELL_11_BLACK,
    BACKGROUND_RING,
    IGNORE = 128,
    CELL_0_WHITE = IGNORE + 1,
    CELL_1_WHITE = IGNORE + 2,
    CELL_2_WHITE = IGNORE + 3,
    CELL_3_WHITE = IGNORE + 4,
    CELL_4_WHITE = IGNORE + 5,
    CELL_5_WHITE = IGNORE + 6,
    CELL_6_WHITE = IGNORE + 7,
    CELL_7_WHITE = IGNORE + 8,
    CELL_8_WHITE = IGNORE + 9,
    CELL_9_WHITE = IGNORE + 10,
    CELL_10_WHITE = IGNORE + 11,
    CELL_11_WHITE = IGNORE + 12,
    OUTER_WHITE_RING = IGNORE + 20,
    INNER_WHITE_SEMICIRCLE = IGNORE + 21
};
__device__ const int MASKS_INDICIES[]= {
    INNER_BLACK_SEMICIRCLE, CELL_0_BLACK, CELL_1_BLACK, CELL_2_BLACK, CELL_3_BLACK,
    CELL_4_BLACK, CELL_5_BLACK, CELL_6_BLACK, CELL_7_BLACK, CELL_8_BLACK, CELL_9_BLACK,
    CELL_10_BLACK, CELL_11_BLACK, BACKGROUND_RING, IGNORE, CELL_0_WHITE, CELL_1_WHITE, CELL_2_WHITE,
    CELL_3_WHITE, CELL_4_WHITE, CELL_5_WHITE, CELL_6_WHITE, CELL_7_WHITE, CELL_8_WHITE,
    CELL_9_WHITE, CELL_10_WHITE, CELL_11_WHITE, OUTER_WHITE_RING, INNER_WHITE_SEMICIRCLE
};



__device__ inline int index2(const int dim1, const int y, const int x) {
    return dim1*y + x;
}

__device__ inline int index3(const int dim2, const int dim1,
               const int z, const int y, const int x) {
    return dim2*dim1*z + dim1*y + x;
}
__device__ inline int index4(const int dim3, const int dim2, const int dim1,
               const int w, const int z, const int y, const int x) {
    return dim3*dim2*dim1*w + dim2*dim1*z + dim1*y + x;
}

template<bool sum_grad_provided, bool pow_grad_provided>
__device__ inline void tmpl_image_mask_split_grad(const float * mask, const float * image,
                                 const float * out_grad_sum, const float * out_grad_pow,
                                 const int bs, const int N, float * grad)
{
    const int b = blockIdx.x;

    const int block_r = blockIdx.y * blockDim.y;
    const int block_c = blockIdx.z * blockDim.z;
    const int r = block_r + threadIdx.y;
    const int c = block_c + threadIdx.x;
    const int sr = r/2;
    const int sc = c/2;

    const int N2 = N/2;
    const int s_idx_base = index4(bs, N2, N2, 0, b, sr, sc);
    const int next_mask_offset = bs*N2*N2;
    const int index = index3(N, N, b, r, c);
    if (b < bs && r < N && c < N && sr < N2 && sc < N2 && index < bs*N*N) {
        float mySum = 0;
        for(int i = 0; i < MASK_LEN; i++) {
            const int s_idx = s_idx_base + i*next_mask_offset;
            if(mask[index] == MASKS_INDICIES[i]) {
                if(sum_grad_provided) {
                    mySum += out_grad_sum[s_idx];
                }
                if (pow_grad_provided) {
                    mySum += 2*image[index]*out_grad_pow[s_idx];
                }
            }
        }
        grad[index] = mySum;
    }
}

extern "C" {

__global__ void to_sum_var_count(const float * reduced, const int n,
                                          float * sum, float * var, float * count) {
    const int block = blockIdx.x * blockDim.x;
    const int tid = threadIdx.x;
    const int idx = block + tid;
    int new_pos = MASK_LEN*floorf(idx / float(3*MASK_LEN)) + idx % MASK_LEN;
    if(idx % 3*MASK_LEN < MASK_LEN) {
        sum[new_pos] = reduced[idx];
    } else if(idx % 3*MASK_LEN < 2*MASK_LEN) {
        new_pos += MASK_LEN;
        var[new_pos] = reduced[idx] - pow(reduced[idx - MASK_LEN], 2);
    } else {
        new_pos += 2*MASK_LEN;
        count[new_pos] = reduced[idx];
    }
}

__global__ void image_mask_split(const float * mask, const float * image,
                                 const int bs, const int N, float * o_split)
{
    const int block_r = blockIdx.y * blockDim.y;
    const int block_c = blockIdx.z * blockDim.z;
    const int sr = block_r + threadIdx.y;
    const int sc = block_c + threadIdx.x;

    const int b = blockIdx.x;
    const int r = 2*sr;
    const int c = 2*sc;
    const int N2 = N/2;
    const int s_idx_base = index4(bs, N2, N2, 0, b, sr, sc);
    const int next_mask_offset = bs*N2*N2;
    const int offset = bs*N2*N2*MASK_LEN;
    if (b < bs && r + 1 < N && c + 1 < N && sr < N2 && sc < N2) {
        for(int j = 0; j <= 1; j++) {
            for(int k = 0; k <=1; k++) {
                int index = index3(N, N, b, r+j, c+k);
                //if (index < bs*N*N) {
                    for(int i = 0; i < MASK_LEN; i++) {
                        const int s_idx = s_idx_base + i*next_mask_offset;
                        if(mask[index] == MASKS_INDICIES[i]) {
                            o_split[s_idx] += image[index];
                            o_split[s_idx + offset] += pow(image[index], 2);
                            o_split[s_idx + 2*offset] += 1;
                        }
                    }
                //}
            }
        }
    }
}
__global__ void image_mask_split_grad_sum_pow(const float * mask, const float * image,
                                 const float * out_grad_sum, const float * out_grad_pow,
                                 const int bs, const int N, float * grad)
{
    tmpl_image_mask_split_grad<true, true>(mask, image, out_grad_sum, out_grad_pow, bs, N, grad);
}

__global__ void image_mask_split_grad_sum(const float * mask, const float * image,
                                          const float * out_grad_sum,  const int bs,
                                          const int N, float * grad)
{
    tmpl_image_mask_split_grad<true, false>(mask, image, out_grad_sum, NULL, bs, N, grad);
}

__global__ void image_mask_split_grad_pow(const float * mask, const float * image,
                                              const float * out_grad_pow, const int bs,
                                              const int N, float * grad)
{
    tmpl_image_mask_split_grad<false, true>(mask, image, NULL, out_grad_pow, bs, N, grad);
}
}
