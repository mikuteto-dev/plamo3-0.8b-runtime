import Foundation
import Metal

final class Batch {
    let scalar: Runtime
    let kv: [(keys:Ref,values:Ref)]
    let hidden: Ref, logits: Ref, result: Ref
    private let x: Ref, qkv: Ref, queries: Ref, attended: Ref, gates: Ref, activated: Ref, scores: Ref
    private let workLogits: Ref, workResult: Ref, positionsBuffer: Ref
    final class Geometry {
        let indices: Ref, counts: Ref, slots: Ref
        var cached = [(lane:Int,first:Int,count:Int)](repeating:(-1,-1,0),count:4)

        init(indices: Ref, counts: Ref, slots: Ref) {
            self.indices = indices; self.counts = counts; self.slots = slots
        }

        func update(_ query: Int, lane: Int, first: Int, count: Int, stride: Int, capacity: Int) {
            let previous = cached[query]
            let start = previous.lane == lane && previous.first == first ? min(previous.count,count) : 0
            let target = indices.buffer.contents().assumingMemoryBound(to:UInt32.self)+query*stride
            for key in start..<count { target[key] = UInt32(lane*capacity+(first+key)%capacity) }
            cached[query] = (lane,first,count)
        }
    }
    private let maps: [Bool:Geometry]
    private let pipelines: [String:MTLComputePipelineState]
    private var normalized: Ref { activated }
    private(set) var positions = [Int]()
    private(set) var ready = false
    private(set) var gpuSeconds = 0.0

    init(_ scalar: Runtime) throws {
        guard (1..<8192).contains(scalar.context), scalar.sparse == nil, scalar.predictor == nil else {
            throw Failure.message("Batch requires dense greedy inference with context below 8192")
        }
        self.scalar = scalar
        let context = scalar.context
        func allocate(_ bytes: Int) throws -> Ref { try allocateBuffer(scalar.device,bytes) }
        kv = try scalar.layers.map { (try allocate(4*$0.capacity*128*2),try allocate(4*$0.capacity*128*2)) }
        x = try allocate(4*1536*4)
        // QKV, attention scores, MLP gates and final logits have disjoint lifetimes.
        let scratch = try allocate(max(4*107520*2,4*16*context*4))
        qkv = scratch; scores = scratch; gates = scratch; workLogits = scratch
        queries = try allocate(4*2048*4)
        attended = queries
        activated = try allocate(4*6144*4)
        hidden = try allocate(4*1536*4)
        logits = try allocate(4*107520*2)
        result = try allocate(4*4)
        workResult = try allocate(4*4)
        positionsBuffer = try allocate(4*4)
        var maps = [Bool:Geometry]()
        for full in [false,true] {
            let indices = try allocate(4*context*4)
            memset(indices.buffer.contents(),0xff,indices.buffer.length)
            maps[full] = Geometry(indices:indices,counts:try allocate(4*4),slots:try allocate(4*4))
        }
        self.maps = maps
        let original = try String(contentsOf:Bundle.module.url(forResource:"Kernels",withExtension:"metal")!,encoding:.utf8)
        let extra = try String(contentsOf:Bundle.module.url(forResource:"Batch",withExtension:"metal")!,encoding:.utf8)
        let options = MTLCompileOptions(); options.fastMathEnabled = false
        let library = try scalar.device.makeLibrary(source:original+"\n"+extra,options:options)
        var pipelines = scalar.pipelines
        for name in ["qkv_mapped","matrix_mapped","softmax_mapped","attend_mapped"] {
            pipelines[name] = try scalar.device.makeComputePipelineState(function:library.makeFunction(name:name)!)
        }
        let formats = Set(([scalar.head]+scalar.layers.flatMap { [$0.qkv,$0.output,$0.up,$0.down] }).map { $0.packing != nil })
        for packed in formats {
            let constants = MTLFunctionConstantValues()
            var flag = packed
            constants.setConstantValue(&flag,type:.bool,index:0)
            for (rows,name) in [(1,"linear"),(2,"linear_pair"),(3,"project3"),(4,"project4")] {
                pipelines["project\(rows)_\(packed)"] = try scalar.device.makeComputePipelineState(function:library.makeFunction(name:name,constantValues:constants))
            }
        }
        self.pipelines = pipelines
    }

    func prefill(_ prompts: [[Int]]) throws -> [Int] {
        guard (1...4).contains(prompts.count), prompts.allSatisfy({ !$0.isEmpty && $0.count <= scalar.context && $0.allSatisfy({ (0..<107520).contains($0) }) }) else {
            throw Failure.message("Expected one to four valid prompts")
        }
        return try autoreleasepool {
            ready = false; gpuSeconds = 0
            for (lane,prompt) in prompts.enumerated() {
                scalar.reset()
                let next = try scalar.prefill(prompt)
                gpuSeconds += scalar.gpuSeconds
                guard let command = scalar.queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder() else { throw Failure.message("Cannot import quad KV") }
                for (index,layer) in scalar.layers.enumerated() {
                    for (source,target) in [(layer.keys,kv[index].keys),(layer.values,kv[index].values)] {
                        blit.copy(from:source.buffer,sourceOffset:source.offset,to:target.buffer,
                                  destinationOffset:target.offset+lane*layer.capacity*128*2,size:min(prompt.count,layer.capacity)*128*2)
                    }
                }
                blit.endEncoding(); command.commit(); command.waitUntilCompleted()
                guard command.status == .completed else { throw Failure.message(command.error?.localizedDescription ?? "Quad KV import failed") }
                gpuSeconds += command.gpuEndTime-command.gpuStartTime
                memcpy(hidden.buffer.contents()+lane*1536*4,scalar.normalized.buffer.contents()+scalar.normalized.offset,1536*4)
                memcpy(logits.buffer.contents()+lane*107520*2,scalar.logits.buffer.contents()+scalar.logits.offset,107520*2)
                result.buffer.contents().assumingMemoryBound(to:UInt32.self)[lane] = UInt32(next)
            }
            positions = prompts.map(\.count); ready = true
            return predictions()
        }
    }

    func step(_ tokens: [Int?]) throws -> [Int] {
        guard ready, tokens.count == positions.count, tokens.contains(where:{ $0 != nil }),
              tokens.compactMap({ $0 }).allSatisfy({ (0..<107520).contains($0) }),
              tokens.indices.allSatisfy({ tokens[$0] == nil || positions[$0] < scalar.context }) else { throw Failure.message("Invalid quad step or context exhausted") }
        let active = tokens.indices.filter { tokens[$0] != nil }, rows = tokens.compactMap { $0 }.count
        return try autoreleasepool {
            for (query,lane) in active.enumerated() { positionsBuffer.buffer.contents().assumingMemoryBound(to:UInt32.self)[query] = UInt32(positions[lane]) }
            var maxima = [Bool:Int]()
            for full in [false,true] {
                let map = maps[full]!, capacity = full ? scalar.context : scalar.layers[0].capacity
                var maximum = 0
                for (query,lane) in active.enumerated() {
                    let pos = positions[lane], first = full ? 0 : max(0,pos-2047), count = pos-first+1
                    maximum = max(maximum,count)
                    map.counts.buffer.contents().assumingMemoryBound(to:UInt32.self)[query] = UInt32(count)
                    map.slots.buffer.contents().assumingMemoryBound(to:UInt32.self)[query] = UInt32(lane*capacity+pos%capacity)
                    map.update(query,lane:lane,first:first,count:count,stride:scalar.context,capacity:capacity)
                }
                maxima[full] = maximum
            }
            guard let command = scalar.queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else { throw Failure.message("Cannot create quad command") }
            scalar.weights.embed(active.map { tokens[$0]! },into:x)
            func run(_ name: String, _ bindings: [(Int,Ref)], _ args: Args, _ width: Int, _ height: Int = 1, _ threads: Int = 256) {
                encoder.setComputePipelineState(pipelines[name]!)
                for (index,ref) in bindings { encoder.setBuffer(ref.buffer,offset:ref.offset,index:index) }
                var a = args
                encoder.setBytes(&a,length:MemoryLayout<Args>.size,index:3)
                encoder.dispatchThreadgroups(MTLSize(width:width,height:height,depth:1),threadsPerThreadgroup:MTLSize(width:threads,height:1,depth:1))
            }
            func norm(_ weight: Ref, _ source: Ref, _ target: Ref, _ offset: Float, residual: Bool = false, next: Ref? = nil) {
                var a = Args(n:1536,m:next == nil ? 0 : 1,batch:UInt32(rows),full:residual ? 1 : 0); a.offset = offset
                run("norm",[(0,weight),(1,source),(2,target),(4,x),(5,next ?? weight),(6,normalized)],a,rows)
            }
            func linear(_ weight: Ref, _ source: Ref, _ target: Ref, _ n: Int, _ m: Int) {
                let offsets = Ref(buffer:weight.buffer,offset:weight.packing?.offsets ?? weight.offset)
                run("project\(rows)_\(weight.packing != nil)",[(0,weight),(1,source),(2,target),(4,offsets)],
                    Args(n:UInt32(n),m:UInt32(m),batch:UInt32(rows),capacity:weight.packing == nil ? 0 : 1,full:m == 107520 ? 2 : 0),(m+7)/8)
            }
            for (index,layer) in scalar.layers.enumerated() {
                if index == 0 { norm(layer.preAttention,x,normalized,1) }
                linear(layer.qkv,normalized,qkv,1536,2304)
                let map = maps[layer.full]!, buffers = kv[index]
                let a = Args(m:UInt32(scalar.context),batch:UInt32(rows),count:UInt32(scalar.context),full:layer.full ? 1 : 0)
                run("qkv_mapped",[(0,layer.qNorm),(1,layer.kNorm),(2,qkv),(4,queries),(5,buffers.keys),(6,buffers.values),(7,scalar.rope),(8,positionsBuffer),(9,map.slots)],a,rows*17,1,32)
                run("matrix_mapped",[(0,buffers.keys),(1,queries),(2,scores),(5,scalar.rope),(6,map.indices),(7,map.counts)],a,(maxima[layer.full]!+31)/32,rows)
                run("softmax_mapped",[(0,scores),(1,map.counts)],a,rows*16)
                run("attend_mapped",[(0,scores),(1,buffers.values),(2,attended),(4,map.indices),(5,map.counts)],a,4,rows)
                linear(layer.output,attended,qkv,2048,1536)
                norm(layer.postAttention,qkv,x,0.2,residual:true,next:layer.preMLP)
                linear(layer.up,normalized,gates,1536,12288)
                run("swiglu",[(0,gates),(1,activated)],Args(batch:UInt32(rows)),(rows*6144+255)/256)
                linear(layer.down,activated,qkv,6144,1536)
                norm(layer.postMLP,qkv,x,Float(1/pow(5.0,1.5)),residual:true,next:index+1 < scalar.layers.count ? scalar.layers[index+1].preAttention : nil)
            }
            norm(scalar.finalNorm,x,normalized,1)
            linear(scalar.head,normalized,workLogits,1536,107520)
            run("argmax",[(0,workLogits),(1,workResult)],Args(n:107520,full:2),rows,1,512)
            encoder.endEncoding()
            guard let blit = command.makeBlitCommandEncoder() else { ready = false; throw Failure.message("Cannot scatter quad outputs") }
            // Packed query rows are temporary; persistent outputs stay indexed by sequence.
            for (query,lane) in active.enumerated() {
                for (source,target,stride) in [(normalized,hidden,1536*4),(workLogits,logits,107520*2),(workResult,result,4)] {
                    blit.copy(from:source.buffer,sourceOffset:source.offset+query*stride,to:target.buffer,destinationOffset:target.offset+lane*stride,size:stride)
                }
            }
            blit.endEncoding(); command.commit(); command.waitUntilCompleted()
            guard command.status == .completed else { ready = false; throw Failure.message(command.error?.localizedDescription ?? "Quad execution failed") }
            gpuSeconds += command.gpuEndTime-command.gpuStartTime
            for lane in active { positions[lane] += 1 }
            return predictions()
        }
    }

    func reset() { ready = false; positions = []; gpuSeconds = 0 }
    func predictions() -> [Int] {
        guard ready else { return [] }
        return positions.indices.map { Int(result.buffer.contents().assumingMemoryBound(to:UInt32.self)[$0]) }
    }
}
