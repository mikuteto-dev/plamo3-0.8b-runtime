import Foundation
import Metal
import Darwin

struct Packing {
    let offsets: Int
}

struct Ref {
    let buffer: MTLBuffer
    var offset: Int = 0
    var packing: Packing? = nil
    func advanced(_ bytes: Int) -> Ref { Ref(buffer: buffer, offset: offset + bytes) }
}

final class Mapping {
    let pointer: UnsafeMutableRawPointer
    let length: Int
    init(_ pointer: UnsafeMutableRawPointer, _ length: Int) { self.pointer = pointer; self.length = length }
    deinit { munmap(pointer, length) }
}

final class Weights {
    let header: [String: Any]
    let gpuBytes: Int
    private let buffers: [(start: Int, end: Int, buffer: MTLBuffer)]
    private let embeddingOffset: Int
    private let mapping: Mapping
    private let dataStart: Int
    private let fileSize: Int
    private let packed: Packed?

    init(path: String, device: MTLDevice, packedPath: String? = nil) throws {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw Failure.message("Cannot open weights: \(path)") }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_size > 8, info.st_size <= 2_000_000_000 else { throw Failure.message("Invalid weights file") }
        fileSize = Int(info.st_size)
        let page = Int(getpagesize())
        let length = (fileSize + page - 1) / page * page
        guard let mapped = mmap(nil, length, PROT_READ, MAP_PRIVATE, fd, 0), mapped != MAP_FAILED else {
            throw Failure.message("Cannot map weights")
        }
        let mapping = Mapping(mapped, length)
        self.mapping = mapping
        let headerSize = mapped.loadUnaligned(as: UInt64.self).littleEndian
        guard headerSize <= 1_048_576, headerSize < fileSize - 8 else {
            throw Failure.message("Invalid safetensors header size")
        }
        dataStart = 8 + Int(headerSize)
        guard dataStart % 8 == 0 else {
            throw Failure.message("Unaligned safetensors header")
        }
        guard let parsed = try JSONSerialization.jsonObject(with: Data(bytes: mapped + 8, count: Int(headerSize))) as? [String: Any] else {
            throw Failure.message("Invalid safetensors header")
        }
        header = parsed
        guard let embedding = parsed["model.embed_tokens.weight"] as? [String: Any],
              embedding["dtype"] as? String == "BF16", embedding["shape"] as? [Int] == [107520, 1536],
              let offsets = embedding["data_offsets"] as? [Int], offsets.count == 2,
              offsets[0] >= 0, offsets[0] % 2 == 0, offsets[1] >= offsets[0], offsets[1] - offsets[0] == 330301440,
              offsets[1] <= fileSize - dataStart else {
            throw Failure.message("Invalid input embedding")
        }
        embeddingOffset = dataStart + offsets[0]
        if let packedPath = packedPath {
            let archive = try Packed(path: packedPath, device: device)
            packed = archive
            buffers = []
            gpuBytes = archive.buffer.length
            return
        }
        packed = nil
        let firstEnd = (embeddingOffset + page - 1) / page * page
        let secondStart = (dataStart + offsets[1]) / page * page
        let release: (UnsafeMutableRawPointer, Int) -> Void = { [mapping] _, _ in withExtendedLifetime(mapping) {} }
        guard firstEnd <= secondStart,
              let first = device.makeBuffer(bytesNoCopy: mapped, length: firstEnd, options: .storageModeShared, deallocator: release),
              let second = device.makeBuffer(bytesNoCopy: mapped + secondStart, length: length - secondStart, options: .storageModeShared, deallocator: release) else {
            throw Failure.message("Metal cannot share the mapped weights")
        }
        buffers = [(0, firstEnd, first), (secondStart, length, second)]
        gpuBytes = firstEnd + length - secondStart
    }

    func tensor(_ name: String, _ shape: [Int]) throws -> Ref {
        guard let t = header[name] as? [String: Any], t["dtype"] as? String == "BF16",
              let actual = t["shape"] as? [Int], actual == shape,
              let offsets = t["data_offsets"] as? [Int], offsets.count == 2,
              offsets[0] >= 0, offsets[0] % 4 == 0, offsets[1] >= offsets[0],
              offsets[1] - offsets[0] == shape.reduce(2, *), offsets[1] <= fileSize - dataStart else {
            throw Failure.message("Missing or incompatible BF16 tensor: \(name)")
        }
        if let packed = packed { return try packed.tensor(name, shape) }
        let start = dataStart + offsets[0], end = dataStart + offsets[1]
        guard let segment = buffers.first(where: { start >= $0.start && end <= $0.end }) else {
            throw Failure.message("GPU tensor overlaps the CPU-only embedding: \(name)")
        }
        return Ref(buffer: segment.buffer, offset: start - segment.start)
    }

    // The untied input embedding needs only the selected rows, not a 330 MB GPU allocation.
    func embed(_ tokens: [Int], into destination: Ref) {
        let source = (mapping.pointer + embeddingOffset).assumingMemoryBound(to: UInt16.self)
        let target = destination.buffer.contents().advanced(by: destination.offset).assumingMemoryBound(to: Float.self)
        for (t, token) in tokens.enumerated() {
            for i in 0..<1536 { target[t*1536+i] = Float(bitPattern: UInt32(source[token*1536+i]) << 16) }
        }
    }
}

struct Args {
    var n: UInt32 = 0, m: UInt32 = 0, batch: UInt32 = 1, start: UInt32 = 0
    var count: UInt32 = 0, capacity: UInt32 = 0, base: UInt32 = 0, full: UInt32 = 0
    var offset: Float = 0
}

struct Layer {
    let preAttention: Ref, postAttention: Ref, preMLP: Ref, postMLP: Ref
    let qkv: Ref, output: Ref, qNorm: Ref, kNorm: Ref, up: Ref, down: Ref
    let keys: Ref, values: Ref
    let capacity: Int
    let full: Bool
    let indexer: Indexer?
}

final class Runtime {
    static let chunk = 16
    static let verifyRows = 2
    private static let splitThreshold = 8192
    let device: MTLDevice
    let weights: Weights
    let queue: MTLCommandQueue
    let pipelines: [String: MTLComputePipelineState]
    let layers: [Layer]
    let head: Ref, finalNorm: Ref, rope: Ref
    private(set) var x: Ref, normalized: Ref, projected: Ref, qkv: Ref
    private(set) var attended: Ref, gates: Ref, activated: Ref, logits: Ref
    let result: Ref
    private(set) var scores: Ref
    private var workBatch = 1
    private let scoreContext: Int
    private(set) var rotatedQueries: Ref
    // Local keys are consumed before the MLP overwrites this scratch.
    var rotatedKeys: Ref { activated }
    let context: Int
    private let packed: Bool
    let batchLimit: Int
    let sparse: Sparse?
    let predictor: Predictor?
    let auxiliaryBytes: Int
    private(set) var lastOutputRows = 0
    private(set) var position = 0
    private(set) var gpuSeconds: Double = 0
    private(set) var peakMetalBytes = 0
    var splitAttention = true
    var specializeWeights = true
    var fuseRotary = true
    private(set) var sparseQueries = 0
    private(set) var sparseKeys = 0
    private(set) var denseKeys = 0

    init(directory: String, context: Int, packed: Bool = false, qsaBudget: Int? = nil, mtp: Bool = false,
         device: MTLDevice? = MTLCreateSystemDefaultDevice()) throws {
        guard (1...32768).contains(context) else { throw Failure.message("Context must be 1...32768") }
        guard let d = device, d.hasUnifiedMemory,
              let q = d.makeCommandQueue() else { throw Failure.message("Apple silicon Metal is required") }
        self.device = d
        self.queue = q
        self.context = context
        self.packed = packed
        batchLimit = min(context,Self.chunk)
        if let budget = qsaBudget, !(4...2048).contains(budget) || budget % 4 != 0 {
            throw Failure.message("QSA budget must be a multiple of four in 4...2048")
        }
        let auxiliary = qsaBudget != nil || mtp ? try Auxiliary(path: directory + "/auxiliary.safetensors", device: d) : nil
        sparse = try qsaBudget.map { try Sparse(d, budget: $0, context: context, chunk: Self.chunk) }
        predictor = try mtp ? Predictor(d, auxiliary!, context: context) : nil
        let config = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: directory + "/config.json"))) as? [String: Any]
        let expected = ["hidden_size": 1536, "num_hidden_layers": 16, "num_attention_heads": 16,
                        "num_key_value_heads": 1, "head_dim": 128, "intermediate_size": 6144,
                        "vocab_size": 107520, "window_size": 2048, "sliding_window_pattern": 8]
        guard let c = config, expected.allSatisfy({ (c[$0.key] as? Int) == $0.value }),
              c["tie_word_embeddings"] as? Bool == false, c["scale_embedding"] as? Bool == false,
              c["model_type"] as? String == "plamo3", c["dtype"] as? String == "bfloat16",
              c["rope_theta"] as? Double == 1_000_000, c["rope_local_theta"] as? Double == 10_000,
              c["rope_scaling_factor"] as? Double == 256, c["rms_norm_eps"] as? Double == 1e-6 else {
            throw Failure.message("Only the original BF16 PLaMo 3 Translate 0.8B configuration is supported")
        }
        let weights = try Weights(path: directory + "/model.safetensors", device: d,
                                  packedPath: packed ? directory + "/model.p12" : nil)
        self.weights = weights
        let source = try String(contentsOf: Bundle.module.url(forResource: "Kernels", withExtension: "metal")!, encoding: .utf8)
        let options = MTLCompileOptions()
        options.fastMathEnabled = false
        let library = try d.makeLibrary(source: source, options: options)
        var p = [String: MTLComputePipelineState]()
        var names = ["norm","linear","matrix","swiglu","qkv","rotate","matrix_rotary","softmax","attend","argmax","init_rope"]
        if mtp { names += ["linear_pair","rotate_steps","mtp_activate","mtp_add"] }
        if let budget = qsaBudget, context > budget {
            names += ["qsa_query","qsa_pool","qsa_scores","qsa_select","qsa_gather"]
        }
        if context >= Self.splitThreshold && qsaBudget == nil && !mtp { names += ["attention","attention_merge"] }
        for name in names {
            let f: MTLFunction
            if ["linear","linear_pair","matrix"].contains(name) {
                f = try library.makeFunction(name:name,constantValues:MTLFunctionConstantValues())
            } else {
                guard let function = library.makeFunction(name:name) else { throw Failure.message("Missing Metal kernel: \(name)") }
                f = function
            }
            p[name] = try d.makeComputePipelineState(function: f)
        }
        let constants = MTLFunctionConstantValues()
        var compressed = packed
        constants.setConstantValue(&compressed,type:.bool,index:0)
        let projections = packed ? (mtp ? ["linear","linear_pair","matrix"] : ["linear","matrix"]) : ["matrix"]
        for name in projections {
            let function = try library.makeFunction(name:name,constantValues:constants)
            p[name+(packed ? "_packed" : "_raw")] = try d.makeComputePipelineState(function:function)
        }
        guard p["argmax"]!.maxTotalThreadsPerThreadgroup >= 512 else {
            throw Failure.message("Metal argmax requires a 512-thread group")
        }
        pipelines = p
        func allocate(_ size: Int) throws -> Ref {
            guard let b = d.makeBuffer(length: size, options: .storageModeShared) else {
                throw Failure.message("Cannot allocate \(size) bytes")
            }
            return Ref(buffer: b)
        }
        head = try weights.tensor("lm_head.weight", [107520, 1536])
        finalNorm = try weights.tensor("model.norm.weight", [1536])
        var ls = [Layer]()
        for i in 0..<16 {
            let prefix = "model.layers.layers.\(i)."
            func w(_ suffix: String, _ shape: [Int]) throws -> Ref { try weights.tensor(prefix + suffix + ".weight", shape) }
            let full = i % 8 == 7
            let capacity = full ? context : min(context, 2048 + batchLimit)
            ls.append(try Layer(
                preAttention: w("pre_mixer_norm", [1536]), postAttention: w("post_mixer_norm", [1536]),
                preMLP: w("pre_mlp_norm", [1536]), postMLP: w("post_mlp_norm", [1536]),
                qkv: w("mixer.qkv_proj", [2304, 1536]), output: w("mixer.o_proj", [1536, 2048]),
                qNorm: w("mixer.q_norm", [128]), kNorm: w("mixer.k_norm", [128]),
                up: w("mlp.gate_up_proj", [12288, 1536]), down: w("mlp.down_proj", [1536, 6144]),
                keys: allocate(capacity * 128 * 2), values: allocate(capacity * 128 * 2), capacity: capacity, full: full,
                indexer: full && qsaBudget != nil && context > qsaBudget! ? Indexer(d, auxiliary!, layer: i, context: context) : nil
            ))
        }
        layers = ls
        auxiliaryBytes = auxiliary?.allocatedBytes ?? 0
        let b = 1
        scoreContext = qsaBudget.map { context > $0 ? max(min(context,2048+Self.chunk),$0+3) : context } ?? context
        x = try allocate(b * 1536 * 4)
        qkv = try allocate(b * 2304 * 4)
        projected = qkv
        rotatedQueries = try allocate(b * 2048 * 4)
        attended = try allocate(b * 2048 * 4)
        gates = try allocate(Self.gateSize(b,mtp ? Self.verifyRows : 1,scoreContext))
        activated = try allocate(b * 6144 * 4)
        normalized = activated
        scores = gates
        logits = gates
        result = try allocate((mtp ? Self.verifyRows : 1)*4)
        let rotaryRows = min(context,2048+Self.chunk)
        rope = try allocate(128*4+rotaryRows*64*4)
        let mlxConfig = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: directory + "/config-mlx.json"))) as? [String: Any]
        guard let frequencies = mlxConfig?["plamo_rope_inv_freq"] as? [String: [Double]],
              let local = frequencies["local"], let full = frequencies["full"], local.count == 64, full.count == 64,
              (local + full).allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw Failure.message("Missing source-exact RoPE frequencies in config-mlx.json")
        }
        (local + full).map(Float.init).withUnsafeBytes { rope.buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        guard let command = q.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
            throw Failure.message("Cannot initialize rotary table")
        }
        encoder.setComputePipelineState(p["init_rope"]!)
        encoder.setBuffer(rope.buffer, offset: 0, index: 0)
        var arguments = Args(count: UInt32(rotaryRows))
        encoder.setBytes(&arguments, length: MemoryLayout<Args>.size, index: 3)
        encoder.dispatchThreadgroups(MTLSize(width: (rotaryRows*64+255)/256, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        guard command.status == .completed else { throw Failure.message(command.error?.localizedDescription ?? "Rotary initialization failed") }
        peakMetalBytes = d.currentAllocatedSize
    }

    private func selectKernel(_ name: String, _ weight: Ref) -> String {
        guard specializeWeights else { return name }
        if weight.packing != nil { return name+"_packed" }
        return !packed && name == "matrix" ? name+"_raw" : name
    }

    private static func gateSize(_ batch: Int, _ rows: Int, _ context: Int) -> Int {
        // Preserve shared score scratch so compact logits do not force an extra buffer.
        let head = rows*107520
        return max(batch*12288*4,max(head*2,min(head*4,batch*16*context*4)))
    }

    private func resizeWork(_ batch: Int, keyBytes: Int = 0) throws {
        let bytes = max(batch*6144*4,keyBytes)
        guard batch != workBatch || activated.buffer.length != bytes else { return }
        let input = try allocateBuffer(device,batch*1536*4)
        let mixed = try allocateBuffer(device,batch*2304*4)
        let queries = try allocateBuffer(device,batch*2048*4)
        let output = try allocateBuffer(device,batch*2048*4)
        let gate = try allocateBuffer(device,Self.gateSize(batch,predictor == nil ? 1 : Self.verifyRows,scoreContext))
        let active = try allocateBuffer(device,bytes)
        if lastOutputRows > 0 {
            memcpy(active.buffer.contents(),normalized.buffer.contents(),lastOutputRows*1536*4)
            memcpy(gate.buffer.contents(),logits.buffer.contents(),lastOutputRows*107520*2)
        }
        peakMetalBytes = max(peakMetalBytes,device.currentAllocatedSize)
        x = input; qkv = mixed; projected = mixed; rotatedQueries = queries; attended = output
        gates = gate; logits = gate; activated = active; normalized = active; scores = gate
        workBatch = batch
    }

    func reset() { position = 0; gpuSeconds = 0; lastOutputRows = 0; sparseQueries = 0; sparseKeys = 0; denseKeys = 0 }

    @discardableResult
    func forward(_ tokens: [Int], output: Bool = true, all: Bool = false, sequential: Bool = false) throws -> Int {
        // Completed Metal command buffers otherwise retain resized work buffers in the outer pool.
        try autoreleasepool { try encode(tokens,output: output,all: all,sequential: sequential) }
    }

    private func encode(_ tokens: [Int], output: Bool, all: Bool, sequential: Bool) throws -> Int {
        guard !tokens.isEmpty, tokens.count <= batchLimit, tokens.allSatisfy({ (0..<107520).contains($0) }),
              (!all || (predictor != nil && tokens.count <= Self.verifyRows)), (!sequential || (predictor != nil && tokens.count <= Self.verifyRows)),
              position + tokens.count <= context else { throw Failure.message("Invalid tokens or context exhausted") }
        let keyBytes = !sequential && (tokens.count > 1 || !fuseRotary)
            ? min(context,2048+Self.chunk)*128*2 : 0
        if tokens.count > workBatch || activated.buffer.length < keyBytes {
            try resizeWork(max(tokens.count,workBatch),keyBytes:keyBytes)
        }
        let count = min(scoreContext,max(position+tokens.count,sparse.map { $0.budget+3 } ?? 0))
        let bytes = splitAttention && tokens.count == 1 && predictor == nil && sparse == nil && count >= Self.splitThreshold
            ? max(16*2064*4,((count+255)/256)*16*(128*4+8)) : min(tokens.count,Self.chunk)*16*count*4
        if bytes > scores.buffer.length {
            let buffer = try allocateBuffer(device,min(min(tokens.count,Self.chunk)*16*scoreContext*4,max(bytes,scores.buffer.length*2)))
            peakMetalBytes = max(peakMetalBytes,device.currentAllocatedSize)
            scores = buffer
        }
        guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
            throw Failure.message("Cannot create Metal command buffer")
        }
        weights.embed(tokens, into: x)
        let batch = tokens.count
        func run(_ name: String, _ bindings: [(Int, Ref)], _ args: Args,
                 _ groups: MTLSize, _ threads: Int = 256) {
            encoder.setComputePipelineState(pipelines[name]!)
            for (index, ref) in bindings { encoder.setBuffer(ref.buffer, offset: ref.offset, index: index) }
            var a = args
            encoder.setBytes(&a, length: MemoryLayout<Args>.size, index: 3)
            encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        }
        func groups(_ n: Int, _ threads: Int = 256) -> MTLSize { MTLSize(width: (n+threads-1)/threads, height: 1, depth: 1) }
        func norm(_ w: Ref, _ source: Ref, _ target: Ref, _ offset: Float, residual: Bool = false, batch n: Int? = nil, next: Ref? = nil) {
            var a = Args(n: 1536, m: next == nil ? 0 : 1, batch: UInt32(n ?? batch)); a.offset = offset; a.full = residual ? 1 : 0
            run("norm", [(0,w),(1,source),(2,target),(4,x),(5,next ?? w),(6,normalized)], a, MTLSize(width: n ?? batch, height: 1, depth: 1))
        }
        func linear(_ w: Ref, _ source: Ref, _ target: Ref, _ n: Int, _ m: Int, batch b: Int? = nil) {
            let count = b ?? batch
            var a = Args(n: UInt32(n), m: UInt32(m), batch: UInt32(count), full: m == 107520 ? 2 : 0)
            a.capacity = w.packing == nil ? 0 : 1
            let offsets = Ref(buffer: w.buffer, offset: w.packing?.offsets ?? w.offset)
            let bindings = [(0,w),(1,source),(2,target),(4,offsets)]
            if count == 1 {
                run(selectKernel("linear",w), bindings, a, groups(m*32))
            } else if sequential && count == Self.verifyRows {
                run(selectKernel("linear_pair",w), bindings, a, groups(m*32))
            } else {
                run(selectKernel("matrix",w), bindings, a, MTLSize(width: (m+31)/32, height: (count+15)/16, depth: 1))
            }
        }
        for (index,layer) in layers.enumerated() {
            if index == 0 { norm(layer.preAttention, x, normalized, 1) }
            if let indexer = layer.indexer, let sparse = sparse {
                linear(indexer.projection, normalized, sparse.projected, 1536, 640)
                run("qsa_query", [(0,indexer.qNorm),(1,sparse.projected),(2,sparse.queries),(4,indexer.rawKeys),(5,rope)],
                    Args(batch: UInt32(batch), start: UInt32(position)), MTLSize(width: batch*5, height: 1, depth: 1), 32)
                let firstBlock = position/4, blocks = (position+batch)/4-firstBlock
                if blocks > 0 {
                    run("qsa_pool", [(0,indexer.kNorm),(1,indexer.rawKeys),(2,indexer.pooledKeys),(4,rope)],
                        Args(base: UInt32(firstBlock)), MTLSize(width: blocks, height: 1, depth: 1), 32)
                }
            }
            linear(layer.qkv, normalized, qkv, 1536, 2304)
            let base = layer.full ? 0 : max(0,position-2047)
            let stepwise = sequential && !layer.full
            let fused = fuseRotary && batch == 1 && !layer.full && !stepwise
            let count = stepwise ? min(position+batch,2048) : position+batch-base
            var a = Args(batch: UInt32(batch),start: UInt32(position),count: UInt32(count),
                         capacity: UInt32(layer.capacity),base: UInt32(base),full: layer.full ? 1 : 0)
            a.offset = stepwise ? 1 : 0
            run("qkv",[(0,layer.qNorm),(1,layer.kNorm),(2,qkv),(4,rotatedQueries),(5,layer.keys),(6,layer.values),(7,rope)],a,
                MTLSize(width: batch*17,height: 1,depth: 1),32)
            var keys = layer.full ? layer.keys : (stepwise ? predictor!.stepKeys : rotatedKeys)
            var values = layer.values
            let selected = layer.indexer != nil && sparse != nil && position+batch > sparse!.budget
            if selected, let indexer = layer.indexer, let sparse = sparse {
                sparseQueries += batch
                for t in 0..<batch {
                    let visible = position+t+1
                    denseKeys += visible
                    sparseKeys += min(visible/4, sparse.budget/4)*4+visible%4
                }
                let slots = sparse.budget+3, blockCount = (position+batch)/4
                let indexArgs = Args(m: UInt32(sparse.budget), batch: UInt32(batch), start: UInt32(position), count: UInt32(blockCount))
                run("qsa_scores", [(0,sparse.queries),(1,indexer.pooledKeys),(2,sparse.scores)], indexArgs,
                    MTLSize(width: blockCount, height: batch, depth: 1), 32)
                run("qsa_select", [(0,sparse.scores),(1,sparse.indices),(2,sparse.counts)], indexArgs,
                    MTLSize(width: batch, height: 1, depth: 1))
                var gather = a; gather.count = UInt32(slots)
                run("qsa_gather", [(0,layer.keys),(1,layer.values),(2,sparse.indices),(4,sparse.counts),(6,sparse.keys),(7,sparse.values)], gather,
                    groups(batch*slots*128))
                keys = sparse.keys; values = sparse.values
                a.n = 1; a.count = UInt32(slots); a.capacity = UInt32(slots); a.offset = 0
            } else if !layer.full && !fused {
                run(stepwise ? "rotate_steps" : "rotate", [(0,layer.keys),(1,keys),(2,rope)], a,
                    groups((stepwise ? batch : 1)*count*128))
            }
            let attentionCount = Int(a.count)
            if splitAttention && batch == 1 && predictor == nil && layer.full && !selected && attentionCount >= Self.splitThreshold {
                a.m = UInt32((attentionCount+255)/256)
                let stats = scores.advanced(Int(a.m)*16*128*4)
                run("attention",[(0,keys),(1,rotatedQueries),(2,scores),(4,values),(5,stats),(6,result)],a,
                    MTLSize(width: Int(a.m),height: 1,depth: 1))
                run("attention_merge",[(0,scores),(1,stats),(2,attended)],a,
                    MTLSize(width: 16,height: 1,depth: 1),128)
            } else {
                if fused {
                    run("matrix_rotary", [(0,layer.keys),(1,rotatedQueries),(2,scores),(5,rope)],
                        Args(m: UInt32(attentionCount), capacity: UInt32(layer.capacity), base: UInt32(base)),
                        MTLSize(width: (attentionCount+31)/32, height: 1, depth: 1))
                } else {
                    run("matrix", [(0,keys),(1,rotatedQueries),(2,scores),(4,keys)],
                        Args(n: 128, m: UInt32(attentionCount), batch: UInt32(batch*16), start: stepwise || selected ? 1 : 0, full: 1),
                        MTLSize(width: (attentionCount+31)/32, height: batch, depth: 1))
                }
                run("softmax", [(0,scores),(1,sparse?.counts ?? result)], a, MTLSize(width: batch*16, height: 1, depth: 1))
                run("attend", [(0,scores),(1,values),(2,attended)], a,
                    MTLSize(width: 4, height: batch, depth: 1))
            }
            linear(layer.output, attended, projected, 2048, 1536)
            norm(layer.postAttention, projected, x, 0.2, residual: true, next: layer.preMLP)
            linear(layer.up, normalized, gates, 1536, 12288)
            run("swiglu", [(0,gates),(1,activated)], Args(batch: UInt32(batch)), groups(batch*6144))
            linear(layer.down, activated, projected, 6144, 1536)
            norm(layer.postMLP, projected, x, Float(1 / pow(5.0, 1.5)), residual: true,
                 next: index+1 < layers.count ? layers[index+1].preAttention : nil)
        }
        let rows = output ? (all ? batch : 1) : 0
        if output {
            norm(finalNorm, x.advanced(all ? 0 : (batch-1)*1536*4), normalized, 1, batch: rows)
            linear(head, normalized, logits, 1536, 107520, batch: rows)
            run("argmax", [(0,logits),(1,result)], Args(n: 107520,full:2), MTLSize(width: rows, height: 1, depth: 1), 512)
        }
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else { throw Failure.message(command.error?.localizedDescription ?? "Metal execution failed") }
        gpuSeconds += command.gpuEndTime - command.gpuStartTime
        position += batch
        lastOutputRows = rows
        return output ? Int(result.buffer.contents().assumingMemoryBound(to: UInt32.self)[rows-1]) : -1
    }

    func prefill(_ tokens: [Int]) throws -> Int {
        try autoreleasepool { try fill(tokens) }
    }

    private func fill(_ tokens: [Int]) throws -> Int {
        guard !tokens.isEmpty, position + tokens.count <= context else { throw Failure.message("Input exceeds context") }
        var next = -1
        for start in stride(from: 0, to: tokens.count, by: batchLimit) {
            let end = min(start + batchLimit, tokens.count)
            next = try forward(Array(tokens[start..<end]), output: end == tokens.count)
        }
        try resizeWork(predictor == nil ? 1 : min(Self.verifyRows,context))
        scores = gates
        return next
    }

    func draft(_ first: Int) throws -> [Int] {
        try autoreleasepool { try predict(first) }
    }

    private func predict(_ first: Int) throws -> [Int] {
        guard let predictor = predictor, position > 0, (0..<107520).contains(first), first != 1, first != 16,
              let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
            throw Failure.message("MTP requires a prefetched context and trained adapters")
        }
        memcpy(predictor.input.buffer.contents(), normalized.buffer.contents(), 1536*4)
        weights.embed([first], into: predictor.input.advanced(1536*4))
        func run(_ name: String, _ bindings: [(Int,Ref)], _ args: Args, _ groups: Int, _ threads: Int = 256) {
            encoder.setComputePipelineState(pipelines[name]!)
            for (i,r) in bindings { encoder.setBuffer(r.buffer, offset: r.offset, index: i) }
            var a = args
            encoder.setBytes(&a, length: MemoryLayout<Args>.size, index: 3)
            encoder.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        }
        func linear(_ weight: Ref, _ input: Ref, _ output: Ref, _ n: Int, _ m: Int) {
            let offsets = Ref(buffer: weight.buffer, offset: weight.packing?.offsets ?? weight.offset)
            let name = selectKernel("linear",weight)
            run(name, [(0,weight),(1,input),(2,output),(4,offsets)],
                Args(n: UInt32(n), m: UInt32(m), capacity: weight.packing == nil ? 0 : 1,full:m == 107520 ? 2 : 0), (m*32+255)/256)
        }
        linear(predictor.up, predictor.input, predictor.activated, 3072, 256)
        run("mtp_activate", [(0,predictor.activated)], Args(), 1)
        linear(predictor.down, predictor.activated, predictor.hidden, 256, 1536)
        run("mtp_add", [(0,predictor.input),(1,predictor.hidden)], Args(), 6)
        linear(head, predictor.hidden, logits, 1536, 107520)
        run("argmax", [(0,logits),(1,result)], Args(n: 107520,full:2), 1, 512)
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        guard command.status == .completed else { throw Failure.message(command.error?.localizedDescription ?? "MTP failed") }
        gpuSeconds += command.gpuEndTime-command.gpuStartTime
        let values = result.buffer.contents().assumingMemoryBound(to: UInt32.self)
        return [Int(values[0])]
    }

    func predictions() -> [Int] {
        let values = result.buffer.contents().assumingMemoryBound(to: UInt32.self)
        return (0..<lastOutputRows).map { Int(values[$0]) }
    }

    func rewind(_ newPosition: Int, hiddenIndex: Int) throws {
        guard hiddenIndex >= 0, hiddenIndex < lastOutputRows,
              newPosition == position-lastOutputRows+hiddenIndex+1 else {
            throw Failure.message("Invalid verification rollback")
        }
        memmove(normalized.buffer.contents(), normalized.buffer.contents()+hiddenIndex*1536*4, 1536*4)
        memmove(logits.buffer.contents(), logits.buffer.contents()+hiddenIndex*107520*2, 107520*2)
        let values = result.buffer.contents().assumingMemoryBound(to: UInt32.self)
        values[0] = values[hiddenIndex]
        position = newPosition
        lastOutputRows = 1
    }

    func logitValues(slot: Int = 0) -> [Float] {
        precondition(slot >= 0 && slot < max(1,lastOutputRows))
        let values = logits.buffer.contents().assumingMemoryBound(to:UInt16.self)+slot*107520
        return (0..<107520).map { Float(bitPattern:UInt32(values[$0])<<16) }
    }

    // Reuses `values` so per-token sampling does not churn a half-megabyte array.
    func logits(into values: inout [Float], slot: Int = 0) {
        precondition(slot >= 0 && slot < max(1,lastOutputRows))
        if values.count != 107520 { values = [Float](repeating: 0, count: 107520) }
        let source = logits.buffer.contents().assumingMemoryBound(to:UInt16.self)+slot*107520
        for i in 0..<107520 { values[i] = Float(bitPattern:UInt32(source[i])<<16) }
    }
}
