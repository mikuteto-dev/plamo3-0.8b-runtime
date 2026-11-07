import XCTest
import Metal
@testable import Plamo

final class LogitsTests: XCTestCase {
    func testStorageMatchesFloat() throws {
        guard FileManager.default.fileExists(atPath:"models/plamo/model.p12") else { throw XCTSkip("Pack weights first") }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw XCTSkip("Metal unavailable") }
        let source = try String(contentsOf:Bundle.module.url(forResource:"Kernels",withExtension:"metal")!)
        let options = MTLCompileOptions();options.fastMathEnabled = false
        let library = try device.makeLibrary(source:source,options:options)
        for packed in [false,true] {
            try autoreleasepool {
                let weights = try Weights(path:"models/plamo/model.safetensors",device:device,packedPath:packed ? "models/plamo/model.p12" : nil)
                let weight = try weights.tensor("lm_head.weight",[107520,1536])
                let constants = MTLFunctionConstantValues()
                var enabled = packed
                constants.setConstantValue(&enabled,type:.bool,index:0)
                for (name,batch) in [("linear",1),("linear_pair",2),("matrix",2),("matrix",15)] {
                    let pipeline = try device.makeComputePipelineState(function:library.makeFunction(name:name,constantValues:constants))
                    let n = 1536, m = 37
                    let values = (0..<batch*n).map { Float(($0*17)%127-63)/64 }
                    let input = try XCTUnwrap(values.withUnsafeBytes { device.makeBuffer(bytes:$0.baseAddress!,length:$0.count) })
                    var expected = [UInt32]()
                    for compact in [false,true] {
                        let output = try XCTUnwrap(device.makeBuffer(length:batch*m*(compact ? 2 : 4)))
                        let command = try XCTUnwrap(queue.makeCommandBuffer()), encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
                        encoder.setComputePipelineState(pipeline)
                        encoder.setBuffer(weight.buffer,offset:weight.offset,index:0)
                        encoder.setBuffer(weight.buffer,offset:weight.packing?.offsets ?? weight.offset,index:4)
                        encoder.setBuffer(input,offset:0,index:1);encoder.setBuffer(output,offset:0,index:2)
                        var args = Args(n:UInt32(n),m:UInt32(m),batch:UInt32(batch),capacity:packed ? 1 : 0,full:compact ? 2 : 0)
                        encoder.setBytes(&args,length:MemoryLayout<Args>.size,index:3)
                        encoder.dispatchThreadgroups(MTLSize(width:name == "matrix" ? (m+31)/32 : (m*32+255)/256,height:1,depth:1),
                                                     threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1))
                        encoder.endEncoding();command.commit();command.waitUntilCompleted()
                        XCTAssertEqual(command.status,.completed)
                        let p = output.contents()
                        let actual = compact ? (0..<batch*m).map { UInt32(p.assumingMemoryBound(to:UInt16.self)[$0])<<16 }
                                             : Array(UnsafeBufferPointer(start:p.assumingMemoryBound(to:UInt32.self),count:batch*m))
                        if expected.isEmpty { expected = actual }
                        XCTAssertEqual(actual,expected,"\(name) packed=\(packed)")
                    }
                }
            }
        }
    }

    func testMatrixRowsAndRewind() throws {
        guard FileManager.default.fileExists(atPath:"models/plamo/auxiliary.safetensors") else { throw XCTSkip("Train adapters first") }
        for packed in [false,true] {
            try autoreleasepool {
                let model = try Runtime(directory:"models/plamo",context:64,packed:packed,mtp:true)
                _ = try model.prefill([1]+Array(500..<532))
                try model.forward([600,700],all:true)
                let ids = model.predictions()
                XCTAssertEqual(ids.count,2)
                let rows = (0..<2).map { model.logitValues(slot:$0) }
                for i in 0..<2 {
                    XCTAssertTrue(rows[i].allSatisfy(\.isFinite))
                    let maximum = try XCTUnwrap(rows[i].max())
                    XCTAssertEqual(ids[i],try XCTUnwrap(rows[i].firstIndex(of:maximum)))
                }
                try model.rewind(model.position,hiddenIndex:1)
                XCTAssertEqual(model.predictions(),[ids[1]])
                XCTAssertEqual(model.logitValues().map(\.bitPattern),rows[1].map(\.bitPattern))
            }
        }
    }

    func testSharedScoreCapacity() throws {
        guard FileManager.default.fileExists(atPath:"models/plamo/model.p12") else { throw XCTSkip("Pack weights first") }
        for context in [64,3359,3360,3361,6719,6720,6721,8192,32768] {
            for mtp in [false,true] {
                try autoreleasepool {
                    let model = try Runtime(directory:"models/plamo",context:context,packed:true,mtp:mtp)
                    _ = try model.prefill([1]+Array(500..<532))
                    let rows = mtp ? 2 : 1, oldHead = rows*107520*4
                    XCTAssertLessThanOrEqual(model.gates.buffer.length,oldHead)
                    XCTAssertGreaterThanOrEqual(model.gates.buffer.length,rows*107520*2)
                    XCTAssertGreaterThanOrEqual(model.gates.buffer.length,min(oldHead,rows*16*context*4))
                    XCTAssertTrue(model.logits.buffer === model.gates.buffer)
                    XCTAssertTrue(model.scores.buffer === model.gates.buffer)
                    for batch in [2,3,8,16] {
                        try model.forward(Array(600..<600+batch))
                        let old = max(batch*12288*4,oldHead)
                        XCTAssertLessThanOrEqual(model.gates.buffer.length,old)
                        XCTAssertGreaterThanOrEqual(model.gates.buffer.length,rows*107520*2)
                        XCTAssertGreaterThanOrEqual(model.gates.buffer.length,min(old,batch*16*context*4))
                    }
                }
            }
        }
    }
}
