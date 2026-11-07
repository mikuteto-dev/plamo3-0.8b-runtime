kernel void qkv_mapped(device const ushort* wq [[buffer(0)]],
                device const ushort* wk [[buffer(1)]],
                device const float* x [[buffer(2)]],
                constant Args& a [[buffer(3)]],
                device float* q [[buffer(4)]],
                device ushort* keys [[buffer(5)]],
                device ushort* values [[buffer(6)]],
                device const float* rope [[buffer(7)]],
                device const uint* positions [[buffer(8)]],
                device const uint* slots [[buffer(9)]],
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
    uint pos=positions[token];
    for (uint j = 0; j < 4; j++) {
        uint d=lane+j*32;
        if (head < 16) {
            uint coordinate=a.full?pos:min(pos,2047u);
            q[token*2048+head*128+d]=rotated(z[j],z[(j+2)%4],d,coordinate,rope,a.full);
        } else {
            uint dst=slots[token]*128+d;
            keys[dst]=bits(a.full?rotated(z[j],z[(j+2)%4],d,pos,rope,true):z[j]);
            values[dst]=bits(x[token*2304+2176+d]);
        }
    }
}


kernel void matrix_mapped(device const ushort* keys [[buffer(0)]],
                          device const float* queries [[buffer(1)]],
                          device float* scores [[buffer(2)]],
                          constant Args& a [[buffer(3)]],
                          device const float* rope [[buffer(5)]],
                          device const uint* indices [[buffer(6)]],
                          device const uint* counts [[buffer(7)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float wt[32*32];
    threadgroup float xt[32*16];
    simdgroup_float8x8 acc(0.0f);
    uint row=group.x*32, query=group.y, visible=counts[query];
    for (uint k=0;k<128;k+=32) {
        for (uint i=tid;i<32*8;i+=256) {
            uint r=row+i/8,c=k+(i%8)*4;
            float4 value=0;
            if (r<visible) {
                uint physical=indices[query*a.m+r]*128;
                for (uint j=0;j<4;j++) {
                    uint d=c+j,other=d<64?d+64:d-64;
                    value[j]=a.full?bf(keys[physical+d]):rotated(bf(keys[physical+d]),bf(keys[physical+other]),d,r,rope,false);
                }
            }
            *reinterpret_cast<threadgroup float4*>(wt+i*4)=value;
        }
        for (uint i=tid;i<32*16;i+=256) {
            uint r=k+i/16,c=i%16;
            xt[i]=queries[query*2048+c*128+r];
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
        if (r<visible) scores[(query*16+c)*a.m+r]=xt[i];
    }
}


kernel void softmax_mapped(device float* scores [[buffer(0)]],
                    device const uint* counts [[buffer(1)]],
                    constant Args& a [[buffer(3)]],
                    uint lane [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]],
                    uint s [[thread_index_in_simdgroup]],
                    uint row [[threadgroup_position_in_grid]]) {
    threadgroup float partial[8];
    threadgroup float maximum;
    device float* p = scores+row*a.count;
    uint visible=counts[row/16], end=visible, begin=0;
    float m = -INFINITY;
    for (uint i = lane; i < visible; i += 256) {
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
    for (uint i = lane; i < visible; i += 256) { float v = exp(p[i]-m); p[i] = v; total += v; }
    total = simd_sum(total);
    if (s == 0) partial[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        total = simd_sum(s < 8 ? partial[s] : 0.0f);
        if (s == 0) partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = partial[0];
    for (uint i = lane; i < visible; i += 256) p[i] /= total;
}


kernel void attend_mapped(device const float* scores [[buffer(0)]],
                   device const ushort* values [[buffer(1)]],
                   device float* out [[buffer(2)]],
                   constant Args& a [[buffer(3)]],
                   device const uint* indices [[buffer(4)]],
                   device const uint* counts [[buffer(5)]],
                   uint tid [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]],
                   uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float vt[32*33];
    threadgroup float pt[32*17];
    simdgroup_float8x8 acc(0.0f);
    uint row = group.x*32, col = group.y*16;
    uint visible=counts[group.y];
    for (uint k = 0; k < visible; k += 32) {
        for (uint i = tid; i < 32*32; i += 256) {
            uint d = i%32, key = k+i/32;
            uint index=key<visible?indices[group.y*a.count+key]*128+row+d:0;
            vt[d*33+i/32] = key < visible ? bf(values[index]) : 0.0f;
        }
        for (uint i = tid; i < 32*16; i += 256) {
            uint h = i/32, key = k+i%32;
            pt[(i%32)*17+h] = key < visible ? scores[(col+h)*a.count+key] : 0.0f;
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


kernel void project3(device const ushort* w [[buffer(0)]],
                        device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                        constant Args& a [[buffer(3)]], device const uint* offsets [[buffer(4)]],
                        uint group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    uint row=group*8+sg;
    if(row>=a.m) return;
    device const ushort4* weights=reinterpret_cast<device const ushort4*>(w+row*a.n);
    device const float4* input=reinterpret_cast<device const float4*>(x);
    float4 accum[3];
    for(uint q=0;q<3;q++) accum[q]=0;
    for(uint i=lane;i<a.n/4;i+=32) {
        float4 value=is_packed(a)?packed_four(w,offsets,row*a.n+i*4):as_type<float4>(uint4(weights[i])<<16);
        #pragma unroll
        for(uint q=0;q<3;q++) accum[q]+=value*input[q*a.n/4+i];
    }
    #pragma unroll
    for(uint q=0;q<3;q++) {
        float sum=simd_sum((accum[q].x+accum[q].y)+(accum[q].z+accum[q].w));
        if(lane==0) {
            if(a.full==2) reinterpret_cast<device ushort*>(y)[q*a.m+row]=bits(sum);
            else y[q*a.m+row]=roundbf(sum);
        }
    }
}

kernel void project4(device const ushort* w [[buffer(0)]],
                        device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                        constant Args& a [[buffer(3)]], device const uint* offsets [[buffer(4)]],
                        uint group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    uint row=group*8+sg;
    if(row>=a.m) return;
    device const ushort4* weights=reinterpret_cast<device const ushort4*>(w+row*a.n);
    device const float4* input=reinterpret_cast<device const float4*>(x);
    float4 accum[4];
    for(uint q=0;q<4;q++) accum[q]=0;
    for(uint i=lane;i<a.n/4;i+=32) {
        float4 value=is_packed(a)?packed_four(w,offsets,row*a.n+i*4):as_type<float4>(uint4(weights[i])<<16);
        #pragma unroll
        for(uint q=0;q<4;q++) accum[q]+=value*input[q*a.n/4+i];
    }
    #pragma unroll
    for(uint q=0;q<4;q++) {
        float sum=simd_sum((accum[q].x+accum[q].y)+(accum[q].z+accum[q].w));
        if(lane==0) {
            if(a.full==2) reinterpret_cast<device ushort*>(y)[q*a.m+row]=bits(sum);
            else y[q*a.m+row]=roundbf(sum);
        }
    }
}
