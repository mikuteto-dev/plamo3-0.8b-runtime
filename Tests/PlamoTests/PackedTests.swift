import XCTest
import Metal
@testable import Plamo

final class PackedTests: XCTestCase {
    func testPackedMatrixMatchesRaw() throws {
        guard FileManager.default.fileExists(atPath: "models/plamo/model.p12") else { throw XCTSkip("Run scripts/pack_weights.py") }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw XCTSkip("Metal unavailable") }
        let raw = try Weights(path: "models/plamo/model.safetensors", device: device)
        let packed = try Weights(path: "models/plamo/model.safetensors", device: device, packedPath: "models/plamo/model.p12")
        XCTAssertLessThan(packed.gpuBytes, raw.gpuBytes)
        let source = try String(contentsOf: Bundle.module.url(forResource: "Kernels", withExtension: "metal")!)
        let options = MTLCompileOptions(); options.fastMathEnabled = false
        let library = try device.makeLibrary(source: source, options: options)
        for (name, shape) in [("lm_head.weight", [107520, 1536]), ("model.layers.layers.0.mixer.qkv_proj.weight", [2304,1536]),
                              ("model.layers.layers.0.mlp.down_proj.weight", [1536,6144])] {
            for batch in [1,3,16,32,64] {
                let input = device.makeBuffer(length: batch*shape[1]*4)!
                let x = input.contents().assumingMemoryBound(to: Float.self)
                for i in 0..<batch*shape[1] { x[i] = Float(i%17-8)/16 }
                var expected = [UInt32]()
                for weights in [raw, packed] {
                    let w = try weights.tensor(name, shape)
                    let output = device.makeBuffer(length: batch*shape[0]*4)!
                    let kernel = batch == 1 ? "linear" : "matrix"
                    let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name:kernel,constantValues:MTLFunctionConstantValues()))
                    let command = queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
                    encoder.setComputePipelineState(pipeline)
                    encoder.setBuffer(w.buffer, offset: w.offset, index: 0)
                    encoder.setBuffer(input, offset: 0, index: 1)
                    encoder.setBuffer(output, offset: 0, index: 2)
                    encoder.setBuffer(w.buffer, offset: w.packing?.offsets ?? w.offset, index: 4)
                    var a = Args(n: UInt32(shape[1]), m: UInt32(shape[0]), batch: UInt32(batch), capacity: w.packing == nil ? 0 : 1)
                    encoder.setBytes(&a, length: MemoryLayout<Args>.size, index: 3)
                    let groups = batch == 1 ? (shape[0]*32+255)/256 : (shape[0]+31)/32
                    let columns = 16
                    encoder.dispatchThreadgroups(MTLSize(width: groups, height: batch == 1 ? 1 : (batch+columns-1)/columns, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                    XCTAssertEqual(command.status, .completed)
                    let actual = Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: UInt32.self), count: batch*shape[0]))
                    if expected.isEmpty { expected = actual } else { XCTAssertEqual(actual, expected) }
                }
            }
        }
    }
}
