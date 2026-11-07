import XCTest
import Metal
@testable import Plamo

final class RotaryMatrixTests: XCTestCase {
    func testFusedRotaryScores() throws {
        guard FileManager.default.fileExists(atPath:"models/plamo/model.p12") else { throw XCTSkip("Pack weights first") }
        let model = try Runtime(directory:"models/plamo",context:4096,packed:true)
        let tune = ProcessInfo.processInfo.environment["PLAMO_ROTARY_TUNE"] == "1"
        var records = [[String:Any]]()
        for (count,capacity,base) in [(1,1,0),(19,19,0),(511,2064,0),(2048,2064,0),(2048,2064,2043),(2048,2064,4111)] {
            autoreleasepool {
                let special: [UInt16] = [0,0x8000,1,0x8001,0x3f80,0xbf80,0x4281,0x0080]
                let keys = (0..<capacity*128).map { i -> UInt16 in
                    if base == 4111 { return special[i%special.count] }
                    if base == 2043 { return UInt16(((i%2)<<15) | (((i*7)%46+95)<<7) | ((i*13)%128)) }
                    return UInt16((Float((i*17)%43-21)/32).bitPattern >> 16)
                }
                let queries = (0..<2048).map { Float(($0*7)%47-23)/32 }
                let k = keys.withUnsafeBytes { model.device.makeBuffer(bytes:$0.baseAddress!,length:$0.count)! }
                let q = queries.withUnsafeBytes { model.device.makeBuffer(bytes:$0.baseAddress!,length:$0.count)! }
                let rotated = model.device.makeBuffer(length:count*128*2)!
                let output = model.device.makeBuffer(length:16*count*4)!
                var expected = [UInt32](), samples = [Bool:[Double]]()
                for round in 0..<(tune ? 12 : 2) {
                    let fused = tune ? [false,true,true,false][round%4] : round == 1
                    let command = model.queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
                    func run(_ name: String, _ bindings: [(Int,MTLBuffer,Int)], _ args: Args, _ groups: MTLSize) {
                        encoder.setComputePipelineState(model.pipelines[name]!)
                        for (i,b,offset) in bindings { encoder.setBuffer(b,offset:offset,index:i) }
                        var a=args
                        encoder.setBytes(&a,length:MemoryLayout<Args>.size,index:3)
                        encoder.dispatchThreadgroups(groups,threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1))
                    }
                    for _ in 0..<(tune ? 8 : 1) {
                        if fused {
                            run("matrix_rotary",[(0,k,0),(1,q,0),(2,output,0),(5,model.rope.buffer,0)],
                                Args(m:UInt32(count),capacity:UInt32(capacity),base:UInt32(base)),
                                MTLSize(width:(count+31)/32,height:1,depth:1))
                        } else {
                            run("rotate",[(0,k,0),(1,rotated,0),(2,model.rope.buffer,0)],
                                Args(count:UInt32(count),capacity:UInt32(capacity),base:UInt32(base)),
                                MTLSize(width:(count*128+255)/256,height:1,depth:1))
                            run("matrix",[(0,rotated,0),(1,q,0),(2,output,0),(4,k,0)],
                                Args(n:128,m:UInt32(count),batch:16,full:1),
                                MTLSize(width:(count+31)/32,height:1,depth:1))
                        }
                    }
                    encoder.endEncoding();command.commit();command.waitUntilCompleted()
                    XCTAssertEqual(command.status,.completed,"\(String(describing:command.error))")
                    let actual=Array(UnsafeBufferPointer(start:output.contents().assumingMemoryBound(to:UInt32.self),count:16*count))
                    if expected.isEmpty { expected=actual }
                    XCTAssertTrue(actual.elementsEqual(expected),"count=\(count) capacity=\(capacity) base=\(base) fused=\(fused)")
                    if round>=4 { samples[fused,default:[]].append((command.gpuEndTime-command.gpuStartTime)/8) }
                }
                if tune {
                    var medians=[String:Double]()
                    for fused in [false,true] {
                        let s=samples[fused]!.sorted()
                        medians[fused ? "fused" : "separate"]=(s[1]+s[2])/2
                    }
                    records.append(["count":count,"capacity":capacity,"base":base,"median_seconds":medians,
                                    "samples":["separate":samples[false]!,"fused":samples[true]!],"bitwise_equal":true])
                    print("ROTARY count=\(count) capacity=\(capacity) base=\(base) medians=\(medians)")
                }
            }
        }
        if tune { try json(records).write(to:URL(fileURLWithPath:"results/rotary-matrix.json"),options:.atomic) }
    }

    func testModelMatchesSeparate() throws {
        guard FileManager.default.fileExists(atPath:"models/plamo/model.p12") else { throw XCTSkip("Pack weights first") }
        let separate=try Runtime(directory:"models/plamo",context:64,packed:true)
        let fused=try Runtime(directory:"models/plamo",context:64,packed:true)
        separate.fuseRotary=false
        let tokens=[1]+Array(500..<532)
        XCTAssertEqual(try separate.prefill(tokens),try fused.prefill(tokens))
        XCTAssertEqual(separate.logitValues().map(\.bitPattern),fused.logitValues().map(\.bitPattern))
        for token in [600,700,800] {
            XCTAssertEqual(try separate.forward([token]),try fused.forward([token]))
            XCTAssertEqual(separate.logitValues().map(\.bitPattern),fused.logitValues().map(\.bitPattern))
        }
    }

    func testModelCosts() throws {
        guard ProcessInfo.processInfo.environment["PLAMO_ROTARY_MODEL_TUNE"] == "1" else { throw XCTSkip("Set PLAMO_ROTARY_MODEL_TUNE=1") }
        guard FileManager.default.fileExists(atPath:"models/plamo/model.p12") else { throw XCTSkip("Pack weights first") }
        for packed in [false,true] {
            for length in [326,4097] {
                try autoreleasepool {
                    let first=try Runtime(directory:"models/plamo",context:length+160,packed:packed)
                    let second=try Runtime(directory:"models/plamo",context:length+160,packed:packed)
                    let reverse=ProcessInfo.processInfo.environment["PLAMO_ROTARY_REVERSE"] == "1"
                    let separate=reverse ? second : first, fused=reverse ? first : second
                    separate.fuseRotary=false
                    let prefix=[1]+(0..<(length-1)).map { 500+$0%37 }
                    let models=reverse ? [fused,separate] : [separate,fused]
                    let next=try models.map { try $0.prefill(prefix) }
                    XCTAssertEqual(next[0],next[1])
                    XCTAssertEqual(separate.logitValues().map(\.bitPattern),fused.logitValues().map(\.bitPattern))
                    var samples=[[String:Any]]()
                    for i in 0..<144 {
                        let order = i%4 == 0 || i%4 == 3 ? [separate,fused] : [fused,separate]
                        for model in order {
                            let gpu=model.gpuSeconds, start=Date.timeIntervalSinceReferenceDate
                            try model.forward([500+i%37])
                            let wall=Date.timeIntervalSinceReferenceDate-start
                            if i>=16 {
                                samples.append(["step":i,"fused":model.fuseRotary,"gpu_seconds":model.gpuSeconds-gpu,"wall_seconds":wall])
                            }
                        }
                        XCTAssertEqual(separate.logitValues().map(\.bitPattern),fused.logitValues().map(\.bitPattern),"packed=\(packed) length=\(length) step=\(i)")
                    }
                    XCTAssertEqual(separate.position,length+144)
                    XCTAssertEqual(fused.position,separate.position)
                    let label="\(packed ? "packed" : "raw")-\(length)\(reverse ? "-reverse" : "")"
                    try json(["packed":packed,"prefix_tokens":length,"samples":samples,"all_logits_bitwise_equal":true,
                              "metal_allocated_bytes_two_models":fused.device.currentAllocatedSize])
                        .write(to:URL(fileURLWithPath:"results/rotary-model-\(label).json"),options:.atomic)
                    print("ROTARY MODEL \(label) passed")
                }
            }
        }
    }
}
