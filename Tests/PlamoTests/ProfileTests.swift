import XCTest
import Metal
@testable import Plamo

final class ProfileTests: XCTestCase {
    func testKernelCosts() throws {
        guard ProcessInfo.processInfo.environment["PLAMO_PROFILE"] == "1" else { throw XCTSkip("Set PLAMO_PROFILE=1") }
        let model = try Runtime(directory: "models/plamo", context: 4096, packed: true, qsaBudget: 128, mtp: true)
        _ = try model.prefill([1]+(0..<325).map { 500+$0%37 })
        _ = try model.draft(500)
        let layer = model.layers[15], sparse = model.sparse!, indexer = layer.indexer!
        func size(_ x: Int, _ y: Int = 1) -> MTLSize { MTLSize(width: x,height: y,depth: 1) }
        func measure(_ label: String, _ name: String, _ bindings: [(Int,Ref)], _ args: Args, _ groups: MTLSize, _ threads: Int = 256) throws {
            var times = [Double]()
            for round in 0..<4 {
                let command = model.queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
                encoder.setComputePipelineState(model.pipelines[name]!)
                for (i,r) in bindings { encoder.setBuffer(r.buffer,offset:r.offset,index:i) }
                var a = args
                encoder.setBytes(&a,length:MemoryLayout<Args>.size,index:3)
                for _ in 0..<8 { encoder.dispatchThreadgroups(groups,threadsPerThreadgroup:size(threads)) }
                encoder.endEncoding();command.commit();command.waitUntilCompleted()
                XCTAssertEqual(command.status,.completed)
                if round > 0 { times.append((command.gpuEndTime-command.gpuStartTime)/8) }
            }
            times.sort()
            print("PROFILE \(label) \(times[times.count/2]*1000) ms")
        }
        func linear(_ label: String, _ w: Ref, _ n: Int, _ m: Int, _ batch: Int = 1) throws {
            let table = Ref(buffer:w.buffer,offset:w.packing?.offsets ?? w.offset)
            let kernel = (batch == 1 ? "linear" : "linear_pair")+(w.packing == nil ? "" : "_packed")
            try measure(label,kernel,[(0,w),(1,model.activated),(2,model.gates),(4,table)],
                        Args(n:UInt32(n),m:UInt32(m),batch:UInt32(batch),capacity:w.packing == nil ? 0 : 1),size((m*32+255)/256))
        }
        try linear("gate-one",layer.up,1536,12288)
        try linear("gate-two",layer.up,1536,12288,2)
        try linear("down-one",layer.down,6144,1536)
        try linear("down-two",layer.down,6144,1536,2)
        // Head output is larger than the scratch used by the other projections.
        for batch in [1,2] {
            let w = model.head
            let table = Ref(buffer:w.buffer,offset:w.packing!.offsets)
            try measure("head-\(batch)",batch == 1 ? "linear_packed" : "linear_pair_packed",[(0,w),(1,model.normalized),(2,model.logits),(4,table)],
                        Args(n:1536,m:107520,batch:UInt32(batch),capacity:1,full:2),size(107520*32/256))
        }
        let matrixInput = try allocateBuffer(model.device,32*6144*4)
        let matrixOutput = try allocateBuffer(model.device,32*12288*4)
        memset(matrixInput.buffer.contents(),0,matrixInput.buffer.length)
        for (label,w,n,m) in [("gate-matrix",layer.up,1536,12288),("down-matrix",layer.down,6144,1536)] {
            let table = Ref(buffer:w.buffer,offset:w.packing!.offsets)
            for batch in [16,32] {
                try measure("\(label)-\(batch)","matrix_packed",[(0,w),(1,matrixInput),(2,matrixOutput),(4,table)],
                            Args(n:UInt32(n),m:UInt32(m),batch:UInt32(batch),capacity:1),size((m+31)/32,(batch+15)/16))
            }
        }
        try linear("indexer",indexer.projection,1536,640)
        let a = Args(m:128,batch:1,start:326,count:81,capacity:4096,full:1)
        try measure("index-query","qsa_query",[(0,indexer.qNorm),(1,sparse.projected),(2,sparse.queries),(4,indexer.rawKeys),(5,model.rope)],a,size(5),32)
        try measure("index-scores","qsa_scores",[(0,sparse.queries),(1,indexer.pooledKeys),(2,sparse.scores)],a,size(81),32)
        try measure("index-select","qsa_select",[(0,sparse.scores),(1,sparse.indices),(2,sparse.counts)],a,size(1))
        var g = a;g.count=131
        try measure("index-gather","qsa_gather",[(0,layer.keys),(1,layer.values),(2,sparse.indices),(4,sparse.counts),(6,sparse.keys),(7,sparse.values)],g,size((131*128+255)/256))
        let attentionScores = try allocateBuffer(model.device,3*16*4096*4)
        for count in [256,2048,4096] {
            for batch in [1,2] {
                let p = attentionScores.buffer.contents().assumingMemoryBound(to: Float.self)
                for i in 0..<batch*16*count { p[i] = 1/Float(count) }
                let a = Args(batch:UInt32(batch),start:UInt32(count-1),count:UInt32(count),capacity:4096,full:1)
                try measure("values-\(count)-b\(batch)","attend",[(0,attentionScores),(1,layer.values),(2,model.attended)],a,size(4,batch))
            }
        }
        let rotated = try allocateBuffer(model.device,327*128*2)
        try measure("local-rotate","rotate",[(0,model.layers[0].keys),(1,rotated),(2,model.rope)],
                    Args(batch:1,start:326,count:327,capacity:2064),size((327*128+255)/256))
    }
}
