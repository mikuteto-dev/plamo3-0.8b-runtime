#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

inline float bf(ushort x) { return as_type<float>(uint(x) << 16); }
inline ushort bits(float x) {
    uint u = as_type<uint>(x);
    return ushort((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
inline float roundbf(float x) { return bf(bits(x)); }

struct Args {
    uint n, m, batch, start, count, capacity, base, full;
    float offset;
};

constant bool packedWeights [[function_constant(0)]];

inline bool is_packed(constant Args& a) {
    return is_function_constant_defined(packedWeights) ? packedWeights : a.capacity != 0;
}

inline ushort packed_bits(device const ushort* w, device const uint* offsets, uint index) {
    uint block = index/128, i = index%128, entry = offsets[block], tag = entry&63u;
    device const uchar* data = reinterpret_cast<device const uchar*>(w)+(entry&~63u);
    uint minimum = tag == 0 ? 0 : tag+107u;
    if (minimum == 0) return reinterpret_cast<device const ushort*>(data)[i];
    uint sm = data[i], code = (data[128+i/2] >> ((i%2)*4)) & 15;
    uint exponent = code == 15 ? 0 : minimum+code;
    return ushort((sm & 127) | ((sm & 128) << 8) | (exponent << 7));
}

inline float4 packed_four(device const ushort* w, device const uint* offsets, uint index) {
    uint block = index/128, i = index%128, entry = offsets[block], tag = entry&63u;
    device const uchar* data = reinterpret_cast<device const uchar*>(w)+(entry&~63u);
    uint minimum = tag == 0 ? 0 : tag+107u;
    if (minimum == 0) return as_type<float4>(uint4(*reinterpret_cast<device const ushort4*>(data+i*2)) << 16);
    ushort4 sm = ushort4(*reinterpret_cast<device const uchar4*>(data+i));
    uint codes = *reinterpret_cast<device const ushort*>(data+128+i/2);
    ushort4 code = (ushort4(codes) >> ushort4(0,4,8,12)) & ushort(15);
    ushort4 exponent = select(ushort4(minimum)+code, ushort4(0), code == 15);
    ushort4 bits = (sm & 127) | ((sm & 128) << 8) | (exponent << 7);
    return as_type<float4>(uint4(bits) << 16);
}

kernel void norm(device const ushort* w [[buffer(0)]],
                 device const float* x [[buffer(1)]],
                 device float* y [[buffer(2)]],
                 device float* residual [[buffer(4)]],
                 device const ushort* nextWeight [[buffer(5)]],
                 device float* next [[buffer(6)]],
                 constant Args& a [[buffer(3)]],
                 uint lane [[thread_index_in_threadgroup]],
                 uint simd [[simdgroup_index_in_threadgroup]],
                 uint s [[thread_index_in_simdgroup]],
                 uint token [[threadgroup_position_in_grid]]) {
    threadgroup float partial[8];
    threadgroup float staged[1536];
    float sum = 0;
    for (uint i = lane; i < a.n; i += 256) sum += x[token*a.n+i] * x[token*a.n+i];
    float v = simd_sum(sum);
    if (s == 0) partial[simd] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        v = simd_sum(s < 8 ? partial[s] : 0.0f);
        if (s == 0) partial[0] = rsqrt(v / float(a.n) + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < a.n; i += 256) {
        float z = roundbf(roundbf(x[token*a.n+i] * partial[0]) * roundbf(bf(w[i]) + a.offset));
        float value = a.full ? roundbf(residual[token*a.n+i] + z) : z;
        y[token*a.n+i] = value;
        if (a.m) staged[i] = value;
    }
    if (!a.m) return;
    // Keep the BF16 boundary and reduction order of the separate next norm.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = 0;
    for (uint i = lane; i < a.n; i += 256) sum += staged[i]*staged[i];
    v = simd_sum(sum);
    if (s == 0) partial[simd] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        v = simd_sum(s < 8 ? partial[s] : 0.0f);
        if (s == 0) partial[0] = rsqrt(v / float(a.n) + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < a.n; i += 256)
        next[token*a.n+i] = roundbf(roundbf(staged[i]*partial[0])*roundbf(bf(nextWeight[i])+1.0f));
}

kernel void linear(device const ushort* w [[buffer(0)]],
                   device const float* x [[buffer(1)]],
                   device float* y [[buffer(2)]],
                   constant Args& a [[buffer(3)]],
                   device const uint* offsets [[buffer(4)]],
                   uint tid [[thread_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]]) {
    uint row = tid / 32;
    if (row >= a.m) return;
    device const ushort4* weights = reinterpret_cast<device const ushort4*>(w + row*a.n);
    device const float4* input = reinterpret_cast<device const float4*>(x);
    float4 accum = 0;
    for (uint i = lane; i < a.n/4; i += 32) {
        float4 v = is_packed(a) ? packed_four(w, offsets, row*a.n+i*4)
                              : as_type<float4>(uint4(weights[i]) << 16);
        accum += v * input[i];
    }
    float sum = simd_sum((accum.x + accum.y) + (accum.z + accum.w));
    if (lane == 0) {
        if (a.full == 2) reinterpret_cast<device ushort*>(y)[row] = bits(sum);
        else y[row] = roundbf(sum);
    }
}

// Preserve scalar accumulation order while reusing weights for two verification tokens.
kernel void linear_pair(device const ushort* w [[buffer(0)]],
                         device const float* x [[buffer(1)]],
                         device float* y [[buffer(2)]],
                         constant Args& a [[buffer(3)]],
                         device const uint* offsets [[buffer(4)]],
                         uint tid [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    uint row = tid/32;
    if (row >= a.m) return;
    device const ushort4* weights = reinterpret_cast<device const ushort4*>(w+row*a.n);
    device const float4* input = reinterpret_cast<device const float4*>(x);
    float4 v0=0, v1=0;
    for (uint i=lane; i<a.n/4; i+=32) {
        float4 v = is_packed(a) ? packed_four(w,offsets,row*a.n+i*4)
                             : as_type<float4>(uint4(weights[i]) << 16);
        v0 += v*input[i];
        v1 += v*input[a.n/4+i];
    }
    float s0=simd_sum((v0.x+v0.y)+(v0.z+v0.w));
    float s1=simd_sum((v1.x+v1.y)+(v1.z+v1.w));
    if (lane == 0) {
        if (a.full == 2) {
            reinterpret_cast<device ushort*>(y)[row]=bits(s0);
            reinterpret_cast<device ushort*>(y)[a.m+row]=bits(s1);
        } else {
            y[row]=roundbf(s0);
            y[a.m+row]=roundbf(s1);
        }
    }
}

// Each SIMD group owns an 8x8 tile; BF16 weights are expanded only in shared tiles.
kernel void matrix(device const ushort* w [[buffer(0)]],
                   device const float* x [[buffer(1)]],
                   device float* y [[buffer(2)]],
                   constant Args& a [[buffer(3)]],
                   device const uint* offsets [[buffer(4)]],
                   uint tid [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]],
                   uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float wt[32*32];
    threadgroup float xt[32*16];
    simdgroup_float8x8 acc(0.0f);
    uint row = group.x * 32;
    uint col = group.y * 16;
    for (uint k = 0; k < a.n; k += 32) {
        for (uint i = tid; i < 32*8; i += 256) {
            uint r = row + i/8, c = k + (i%8)*4;
            uint index = (group.y*a.start*a.m+r)*a.n+c;
            float4 value = 0;
            if (r < a.m && c+3 < a.n) {
                value = is_packed(a) ? packed_four(w, offsets, index)
                                   : as_type<float4>(uint4(*reinterpret_cast<device const ushort4*>(w+index)) << 16);
            } else if (r < a.m) {
                for (uint j = 0; j < 4 && c+j < a.n; j++)
                    value[j] = bf(is_packed(a) ? packed_bits(w, offsets, index+j) : w[index+j]);
            }
            *reinterpret_cast<threadgroup float4*>(wt+i*4) = value;
        }
        for (uint i = tid; i < 32*16; i += 256) {
            uint r = k + i/16, c = col + i%16;
            xt[i] = r < a.n && c < a.batch ? x[c*a.n+r] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < 32; j += 8) {
            simdgroup_float8x8 aw, ax;
            simdgroup_load(aw, wt + (sg/2)*8*32 + j, 32);
            simdgroup_load(ax, xt + j*16 + (sg%2)*8, 16);
            simdgroup_multiply_accumulate(acc, aw, ax, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // The final barrier ends all tile reads, so output can reuse the input tile.
    simdgroup_store(acc, xt + (sg/2)*8*16 + (sg%2)*8, 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 32*16; i += 256) {
        uint r = row + i/16, c = col + i%16;
        float value=xt[i];
        if (r < a.m && c < a.batch) {
            if (a.full == 2) reinterpret_cast<device ushort*>(y)[c*a.m+r] = bits(value);
            else y[c*a.m+r] = a.full ? value : roundbf(value);
        }
    }
}

kernel void swiglu(device const float* x [[buffer(0)]],
                   device float* y [[buffer(1)]],
                   constant Args& a [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
    if (i >= a.batch*6144) return;
    uint p = (i/6144)*12288 + i%6144;
    float g = x[p];
    y[i] = roundbf(roundbf(g / (1.0f + exp(-g))) * x[p+6144]);
}

kernel void init_rope(device float* frequencies [[buffer(0)]],
                      constant Args& a [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if(i>=a.count*64)return;
    float phase=float(i/64)*frequencies[i%64];
    device ushort2* table=reinterpret_cast<device ushort2*>(frequencies+128);
    table[i]=ushort2(bits(cos(phase)),bits(sin(phase)));
}

inline float rotated(float x, float other, uint d, uint pos, device const float* rope, bool full);

kernel void qkv(device const ushort* wq [[buffer(0)]],
                device const ushort* wk [[buffer(1)]],
                device const float* x [[buffer(2)]],
                constant Args& a [[buffer(3)]],
                device float* q [[buffer(4)]],
                device ushort* keys [[buffer(5)]],
                device ushort* values [[buffer(6)]],
                device const float* rope [[buffer(7)]],
                uint lane [[thread_index_in_simdgroup]],
                uint group [[threadgroup_position_in_grid]]) {
    uint token = group/17, head = group%17;
    uint src = token*2304+head*128;
    float v[4];
    float sum = 0;
    for (uint j = 0; j < 4; j++) {
        v[j] = x[src+lane+j*32];
        sum += v[j]*v[j];
    }
    float scale = rsqrt(simd_sum(sum)/128.0f+1e-6f);
    float z[4];
    for(uint j=0;j<4;j++) {
        uint d=lane+j*32;
        z[j]=roundbf(roundbf(v[j]*scale)*roundbf(bf(head<16?wq[d]:wk[d])+1.0f));
    }
    uint pos=a.start+token;
    for (uint j = 0; j < 4; j++) {
        uint d=lane+j*32;
        if (head < 16) {
            uint coordinate=a.full?pos:(a.offset!=0?min(pos,2047u):pos-a.base);
            q[token*2048+head*128+d]=rotated(z[j],z[(j+2)%4],d,coordinate,rope,a.full);
        } else {
            uint dst=(pos%a.capacity)*128+d;
            keys[dst]=bits(a.full?rotated(z[j],z[(j+2)%4],d,pos,rope,true):z[j]);
            values[dst]=bits(x[token*2304+2176+d]);
        }
    }
}

inline float rotated(float x, float other, uint d, uint pos,
                     device const float* rope, bool full) {
    float c,s;
    if(full) {
        float angle=float(pos)*rope[64+d%64];
        c=roundbf(roundbf(cos(angle))*1.5545177444479562f);
        s=roundbf(roundbf(sin(angle))*1.5545177444479562f);
    } else {
        device const ushort2* table=reinterpret_cast<device const ushort2*>(rope+128);
        ushort2 value=table[pos*64+d%64];
        c=bf(value.x);s=bf(value.y);
    }
    return roundbf(roundbf(x*c) + roundbf((d < 64 ? -other : other)*s));
}

// Rebase raw local keys exactly as the source cache does after cropping.
kernel void rotate(device const ushort* keys [[buffer(0)]],
                   device ushort* rotatedKeys [[buffer(1)]],
                   device const float* rope [[buffer(2)]],
                   constant Args& a [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
    if(i>=a.count*128)return;
    uint d=i%128,other=d<64?d+64:d-64,pos=i/128,base=((a.base+pos)%a.capacity)*128;
    rotatedKeys[i]=bits(rotated(bf(keys[base+d]),bf(keys[base+other]),d,pos,rope,false));
}

// Each local verification query uses the same cropped coordinates as a scalar decode.
kernel void rotate_steps(device const ushort* keys [[buffer(0)]],
                         device ushort* rotatedKeys [[buffer(1)]],
                         device const float* rope [[buffer(2)]],
                         constant Args& a [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if(i>=a.batch*a.count*128)return;
    uint d=i%128,other=d<64?d+64:d-64,token=i/(a.count*128),key=(i/128)%a.count;
    uint pos=a.start+token,base=pos>2047?pos-2047:0,index=((base+key)%a.capacity)*128;
    rotatedKeys[i]=key<=min(pos,2047u)
        ?bits(rotated(bf(keys[index+d]),bf(keys[index+other]),d,key,rope,false)):0;
}

// Prefill would repeat this rotation per query tile, so this path is scalar-only.
kernel void matrix_rotary(device const ushort* keys [[buffer(0)]],
                          device const float* queries [[buffer(1)]],
                          device float* scores [[buffer(2)]],
                          constant Args& a [[buffer(3)]],
                          device const float* rope [[buffer(5)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float wt[32*32];
    threadgroup float xt[32*16];
    simdgroup_float8x8 acc(0.0f);
    uint row=group.x*32;
    for (uint k=0;k<128;k+=32) {
        for (uint i=tid;i<32*8;i+=256) {
            uint r=row+i/8,c=k+(i%8)*4;
            float4 value=0;
            if (r<a.m) {
                uint physical=((a.base+r)%a.capacity)*128;
                for (uint j=0;j<4;j++) {
                    uint d=c+j,other=d<64?d+64:d-64;
                    value[j]=rotated(bf(keys[physical+d]),bf(keys[physical+other]),d,r,rope,false);
                }
            }
            *reinterpret_cast<threadgroup float4*>(wt+i*4)=value;
        }
        for (uint i=tid;i<32*16;i+=256) {
            uint r=k+i/16,c=i%16;
            xt[i]=queries[c*128+r];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j=0;j<32;j+=8) {
            simdgroup_float8x8 aw,ax;
            simdgroup_load(aw,wt+(sg/2)*8*32+j,32);
            simdgroup_load(ax,xt+j*16+(sg%2)*8,16);
            simdgroup_multiply_accumulate(acc,aw,ax,acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc,xt+(sg/2)*8*16+(sg%2)*8,16);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=tid;i<32*16;i+=256) {
        uint r=row+i/16,c=i%16;
        if (r<a.m) scores[c*a.m+r]=xt[i];
    }
}

kernel void mtp_activate(device float* x [[buffer(0)]],
                         uint i [[thread_position_in_grid]]) {
    if (i<256) { float v=x[i]; x[i]=roundbf(v/(1.0f+exp(-v))); }
}

kernel void mtp_add(device const float* x [[buffer(0)]],
                    device float* y [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
    if (i<1536) y[i]=roundbf(x[i]+y[i]);
}

kernel void softmax(device float* scores [[buffer(0)]],
                    device const uint* counts [[buffer(1)]],
                    constant Args& a [[buffer(3)]],
                    uint lane [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]],
                    uint s [[thread_index_in_simdgroup]],
                    uint row [[threadgroup_position_in_grid]]) {
    threadgroup float partial[8];
    threadgroup float maximum;
    device float* p = scores+row*a.count;
    uint end = a.n ? counts[row/16] : (a.offset != 0 ? min(a.start+row/16+1,a.count) : a.start+row/16-a.base+1);
    uint begin = a.n || a.full || a.offset != 0 ? 0 : (end > 2048 ? end-2048 : 0);
    float m = -INFINITY;
    for (uint i = lane; i < a.count; i += 256) {
        float v = i >= begin && i < end ? p[i] * 0.08838834764831845f : -INFINITY;
        p[i] = v;
        m = max(m,v);
    }
    m = simd_max(m);
    if (s == 0) partial[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        m = simd_max(s < 8 ? partial[s] : -INFINITY);
        if (s == 0) maximum = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = maximum;
    float total = 0;
    for (uint i = lane; i < a.count; i += 256) { float v = exp(p[i]-m); p[i] = v; total += v; }
    total = simd_sum(total);
    if (s == 0) partial[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        total = simd_sum(s < 8 ? partial[s] : 0.0f);
        if (s == 0) partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = partial[0];
    for (uint i = lane; i < a.count; i += 256) p[i] /= total;
}

// Odd shared strides avoid bank conflicts during the MQA tile transposes.
kernel void attend(device const float* scores [[buffer(0)]],
                   device const ushort* values [[buffer(1)]],
                   device float* out [[buffer(2)]],
                   constant Args& a [[buffer(3)]],
                   uint tid [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]],
                   uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float vt[32*33];
    threadgroup float pt[32*17];
    simdgroup_float8x8 acc(0.0f);
    uint row = group.x*32, col = group.y*16;
    uint valueBase = a.offset != 0 && !a.full ? (a.start+group.y>2047?a.start+group.y-2047:0) : a.base;
    valueBase %= a.capacity;
    for (uint k = 0; k < a.count; k += 32) {
        for (uint i = tid; i < 32*32; i += 256) {
            uint d = i%32, key = k+i/32;
            // Visible keys fit the cache, so a normalized base wraps at most once.
            uint physical = valueBase+key;
            uint index = a.n ? (group.y*a.count+key)*128+row+d : (physical >= a.capacity ? physical-a.capacity : physical)*128+row+d;
            vt[d*33+i/32] = key < a.count ? bf(values[index]) : 0.0f;
        }
        for (uint i = tid; i < 32*16; i += 256) {
            uint h = i/32, key = k+i%32;
            pt[(i%32)*17+h] = key < a.count ? scores[(col+h)*a.count+key] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < 32; j += 8) {
            simdgroup_float8x8 v, p;
            simdgroup_load(v, vt+(sg/2)*8*33+j, 33);
            simdgroup_load(p, pt+j*17+(sg%2)*8, 17);
            simdgroup_multiply_accumulate(acc, v, p, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc, pt+(sg/2)*8*17+(sg%2)*8, 17);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 32*16; i += 256) out[(col+i%16)*128+row+i/16] = roundbf(pt[(i/16)*17+i%16]);
}

// Keep each 256-key partition on chip; only partial outputs reach device memory.
kernel void attention(device const ushort* keys [[buffer(0)]],
                      device const float* queries [[buffer(1)]],
                      device float* out [[buffer(2)]],
                      constant Args& a [[buffer(3)]],
                      device const ushort* values [[buffer(4)]],
                      device float2* stats [[buffer(5)]],
                      device const uint* counts [[buffer(6)]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float wt[32*33], xt[32*17], scores[16*256], totals[16];
    uint token=group.y, headBase=0, first=group.x*256;
    uint length=min(256u,a.count-first);
    bool separate=a.n || (a.offset!=0 && !a.full);
    uint end=a.n?counts[token]:(a.offset!=0?min(a.start+token+1,a.count):a.start+token-a.base+1);
    uint begin=a.n || a.full || a.offset!=0?0:(end>2048?end-2048:0);
    uint valueBase=a.offset!=0 && !a.full?(a.start+token>2047?a.start+token-2047:0):a.base;
    for(uint tile=0;tile<length;tile+=32) {
        simdgroup_float8x8 acc(0.0f);
        for(uint k=0;k<128;k+=32) {
            for(uint i=tid;i<32*32;i+=256) {
                uint key=first+tile+i/32, d=k+i%32;
                uint index=((separate?token*a.count:0)+key)*128+d;
                wt[i]=key<a.count?bf(keys[index]):0.0f;
            }
            for(uint i=tid;i<32*16;i+=256)
                xt[(i/16)*17+i%16]=queries[token*2048+(i%16)*128+k+i/16];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint j=0;j<32;j+=8) {
                simdgroup_float8x8 w,q;
                simdgroup_load(w,wt+(sg/2)*8*32+j,32);
                simdgroup_load(q,xt+j*17+(sg%2)*8,17);
                simdgroup_multiply_accumulate(acc,w,q,acc);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        simdgroup_store(acc,xt+(sg/2)*8*17+(sg%2)*8,17);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=tid;i<32*16;i+=256) {
            uint key=first+tile+i/16;
            scores[(i%16)*256+tile+i/16]=key>=begin && key<end && key<a.count
                ?xt[(i/16)*17+i%16]*0.08838834764831845f:-INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint h=tid/16, lane=tid%16;
    float maximum=-INFINITY;
    for(uint i=lane;i<length;i+=16) maximum=max(maximum,scores[h*256+i]);
    for(ushort delta=8;delta>0;delta/=2) maximum=max(maximum,simd_shuffle_xor(maximum,delta));
    float total=0;
    for(uint i=lane;i<256;i+=16) {
        float value=i<length && isfinite(maximum)?exp(scores[h*256+i]-maximum):0.0f;
        scores[h*256+i]=value; total+=value;
    }
    for(ushort delta=8;delta>0;delta/=2) total+=simd_shuffle_xor(total,delta);
    if(lane==0) {
        totals[h]=total;
        if(a.m>1) stats[(token*a.m+group.x)*16+headBase+h]=float2(maximum,total);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint row=0;row<128;row+=32) {
        simdgroup_float8x8 acc(0.0f);
        for(uint k=0;k<length;k+=32) {
            for(uint i=tid;i<32*32;i+=256) {
                uint key=first+k+i/32, d=row+i%32;
                uint index=a.n?(token*a.count+key)*128+d:((valueBase+key)%a.capacity)*128+d;
                wt[(i%32)*33+i/32]=key<a.count?bf(values[index]):0.0f;
            }
            for(uint i=tid;i<32*16;i+=256) xt[(i%32)*17+i/32]=scores[(i/32)*256+k+i%32];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint j=0;j<32;j+=8) {
                simdgroup_float8x8 v,p;
                simdgroup_load(v,wt+(sg/2)*8*33+j,33);
                simdgroup_load(p,xt+j*17+(sg%2)*8,17);
                simdgroup_multiply_accumulate(acc,v,p,acc);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        simdgroup_store(acc,xt+(sg/2)*8*17+(sg%2)*8,17);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=tid;i<32*16;i+=256) {
            uint head=i%16, d=row+i/16;
            float value=xt[(i/16)*17+i%16];
            if(a.m==1) out[(token*16+head)*128+d]=roundbf(value/totals[i%16]);
            else out[((token*a.m+group.x)*16+head)*128+d]=value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void attention_merge(device const float* parts [[buffer(0)]],
                            device const float2* stats [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            constant Args& a [[buffer(3)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint row [[threadgroup_position_in_grid]]) {
    threadgroup float partial[4], ratios[128], maximum, total;
    uint token=row/16, head=row%16;
    float2 value=tid<a.m?stats[(token*a.m+tid)*16+head]:float2(-INFINITY,0);
    float m=simd_max(value.x);
    if(lane==0) partial[sg]=m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0) {
        m=simd_max(lane<4?partial[lane]:-INFINITY);
        if(lane==0) maximum=m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float weight=tid<a.m && value.y>0?exp(value.x-maximum):0.0f;
    if(tid<a.m) ratios[tid]=weight;
    float sum=simd_sum(weight*value.y);
    if(lane==0) partial[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0) {
        sum=simd_sum(lane<4?partial[lane]:0.0f);
        if(lane==0) total=sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum=0;
    for(uint i=0;i<a.m;i++) sum+=parts[((token*a.m+i)*16+head)*128+tid]*ratios[i];
    out[row*128+tid]=roundbf(sum/total);
}

kernel void qsa_query(device const ushort* weight [[buffer(0)]],
                      device const float* x [[buffer(1)]],
                      device float* queries [[buffer(2)]],
                      constant Args& a [[buffer(3)]],
                      device ushort* rawKeys [[buffer(4)]],
                      device const float* rope [[buffer(5)]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint group [[threadgroup_position_in_grid]]) {
    uint token=group/5, head=group%5;
    if (head==4) {
        for (uint j=0;j<4;j++) rawKeys[((a.start+token)&31)*128+lane+j*32]=bits(x[token*640+512+lane+j*32]);
        return;
    }
    float v[4], z[4], sum=0;
    for (uint j=0;j<4;j++) { v[j]=x[token*640+head*128+lane+j*32]; sum+=v[j]*v[j]; }
    float scale=rsqrt(simd_sum(sum)/128.0f+1e-6f);
    for (uint j=0;j<4;j++) z[j]=roundbf(roundbf(v[j]*scale)*roundbf(bf(weight[lane+j*32])+1.0f));
    for (uint j=0;j<4;j++) queries[token*512+head*128+lane+j*32]=rotated(z[j],z[(j+2)%4],lane+j*32,a.start+token,rope,true);
}

kernel void qsa_pool(device const ushort* weight [[buffer(0)]],
                     device const ushort* rawKeys [[buffer(1)]],
                     device ushort* pooled [[buffer(2)]],
                     constant Args& a [[buffer(3)]],
                     device const float* rope [[buffer(4)]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint group [[threadgroup_position_in_grid]]) {
    uint block=a.base+group;
    float v[4],z[4],sum=0;
    for (uint j=0;j<4;j++) {
        uint d=lane+j*32, p=((block*4)&31)*128+d;
        v[j]=roundbf(((bf(rawKeys[p])+bf(rawKeys[p+128]))+bf(rawKeys[p+256])+bf(rawKeys[p+384]))/4.0f);
        sum+=v[j]*v[j];
    }
    float scale=rsqrt(simd_sum(sum)/128.0f+1e-6f);
    for (uint j=0;j<4;j++) z[j]=roundbf(roundbf(v[j]*scale)*roundbf(bf(weight[lane+j*32])+1.0f));
    for (uint j=0;j<4;j++) pooled[block*128+lane+j*32]=bits(rotated(z[j],z[(j+2)%4],lane+j*32,block*4,rope,true));
}

kernel void qsa_scores(device const float* queries [[buffer(0)]],
                       device const ushort* keys [[buffer(1)]],
                       device float* scores [[buffer(2)]],
                       constant Args& a [[buffer(3)]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint2 group [[threadgroup_position_in_grid]]) {
    uint block=group.x, token=group.y;
    if ((block+1)*4>a.start+token+1) { if(lane==0)scores[token*a.count+block]=-INFINITY; return; }
    float score=0;
    for (uint h=0;h<4;h++) {
        float sum=0;
        for (uint j=0;j<4;j++) { uint d=lane+j*32; sum+=queries[token*512+h*128+d]*bf(keys[block*128+d]); }
        sum=simd_sum(sum); score+=max(sum,0.0f);
    }
    if(lane==0)scores[token*a.count+block]=score*0.08838834764831845f;
}

// Radix threshold selection avoids sorting or copying the entire visible KV cache.
kernel void qsa_select(device const float* scores [[buffer(0)]],
                       device uint* indices [[buffer(1)]],
                       device uint* counts [[buffer(2)]],
                       constant Args& a [[buffer(3)]],
                       uint lane [[thread_index_in_threadgroup]],
                       uint token [[threadgroup_position_in_grid]]) {
    uint visible=a.start+token+1, complete=visible/4, tail=visible%4;
    uint chosen=min(complete,a.m/4), output=token*(a.m+3);
    if(lane==0)counts[token]=chosen*4+tail;
    if(chosen==complete) {
        for(uint i=lane;i<visible;i+=256)indices[output+i]=i;
        return;
    }
    threadgroup atomic_uint histogram[256];
    threadgroup uint prefix, mask, rank;
    threadgroup uint above[256],equal[256],starts[256],tiePrefix[256];
    threadgroup uint allowance;
    if(lane==0) {prefix=0;mask=0;rank=chosen;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(int shift=24;shift>=0;shift-=8) {
        atomic_store_explicit(&histogram[lane],0,memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=lane;i<complete;i+=256) {
            float s=scores[token*a.count+i]; uint value=as_type<uint>(s>0?s:0.0f);
            if((value&mask)==prefix)atomic_fetch_add_explicit(&histogram[(value>>shift)&255],1,memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(lane==0) {
            uint remaining=rank;
            for(int b=255;b>=0;b--) {
                uint count=atomic_load_explicit(&histogram[b],memory_order_relaxed);
                if(remaining>count)remaining-=count;
                else {prefix|=uint(b)<<shift;rank=remaining;break;}
            }
            mask|=255u<<shift;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint chunk=(complete+255)/256,begin=lane*chunk,end=min(begin+chunk,complete),hi=0,eq=0;
    for(uint i=begin;i<end;i++) {
        float s=scores[token*a.count+i];uint value=as_type<uint>(s>0?s:0.0f);
        hi+=value>prefix;eq+=value==prefix;
    }
    above[lane]=hi;equal[lane]=eq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane==0) {
        uint total=0,ties=0;
        for(uint i=0;i<256;i++){total+=above[i];tiePrefix[i]=ties;ties+=equal[i];}
        allowance=chosen-total;
        uint out=0;
        for(uint i=0;i<256;i++) {
            starts[i]=out;
            out+=above[i]+min(equal[i],allowance>tiePrefix[i]?allowance-tiePrefix[i]:0u);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint out=starts[lane],seen=0;
    for(uint i=begin;i<end;i++) {
        float s=scores[token*a.count+i];uint value=as_type<uint>(s>0?s:0.0f);
        bool take=value>prefix;
        if(value==prefix){take=tiePrefix[lane]+seen<allowance;seen++;}
        if(take){for(uint j=0;j<4;j++)indices[output+out*4+j]=i*4+j;out++;}
    }
    if(lane<tail)indices[output+chosen*4+lane]=complete*4+lane;
}

kernel void qsa_gather(device const ushort* keys [[buffer(0)]],
                       device const ushort* values [[buffer(1)]],
                       device const uint* indices [[buffer(2)]],
                       constant Args& a [[buffer(3)]],
                       device const uint* counts [[buffer(4)]],
                       device ushort* selectedKeys [[buffer(6)]],
                       device ushort* selectedValues [[buffer(7)]],
                       uint i [[thread_position_in_grid]]) {
    if(i>=a.batch*a.count*128)return;
    uint token=i/(a.count*128), key=(i/128)%a.count,d=i%128;
    if(key>=counts[token]){selectedKeys[i]=0;selectedValues[i]=0;return;}
    uint position=indices[token*a.count+key];
    if(position>=a.capacity || position>a.start+token){selectedKeys[i]=0;selectedValues[i]=0;return;}
    selectedKeys[i]=keys[position*128+d];
    selectedValues[i]=values[position*128+d];
}

kernel void argmax(device const float* x [[buffer(0)]],
                   device uint* out [[buffer(1)]],
                   constant Args& a [[buffer(3)]],
                   uint lane [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]],
                   uint s [[thread_index_in_simdgroup]],
                   uint row [[threadgroup_position_in_grid]]) {
    threadgroup float maxima[16];
    threadgroup uint indices[16];
    float best = -INFINITY; uint idx = 0xffffffff;
    for (uint i = lane; i < a.n; i += 512) {
        float v = a.full == 2 ? bf(reinterpret_cast<device const ushort*>(x)[row*a.n+i]) : x[row*a.n+i];
        if (v > best || (v == best && i < idx)) { best=v; idx=i; }
    }
    float m = simd_max(best);
    idx = simd_min(best == m ? idx : 0xffffffff);
    if (s == 0) { maxima[sg]=m; indices[sg]=idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        best = s < 16 ? maxima[s] : -INFINITY;
        idx = s < 16 ? indices[s] : 0xffffffff;
        m = simd_max(best);
        idx = simd_min(best == m ? idx : 0xffffffff);
        if (s == 0) out[row]=idx;
    }
}
