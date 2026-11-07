import XCTest
import Metal
@testable import Plamo

final class TuningTests: XCTestCase {
    func testGEMVWidths() throws {
        guard ProcessInfo.processInfo.environment["PLAMO_TUNE"] == "1" else { throw XCTSkip("Set PLAMO_TUNE=1 for M2 calibration") }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw XCTSkip("Metal unavailable") }
        let packed = ProcessInfo.processInfo.environment["PLAMO_TUNE_PACKED"] == "1"
        let weights = try Weights(path: "models/plamo/model.safetensors", device: device,
                                  packedPath: packed ? "models/plamo/model.p12" : nil)
        let source = try String(contentsOf: Bundle.module.url(forResource: "Kernels", withExtension: "metal")!)
        let range = source.range(of: "kernel void linear")!.lowerBound..<source.range(of: "// Each SIMD group")!.lowerBound
        let original = String(source[range])
        var pipelines = [Int: MTLComputePipelineState]()
        for width in [4, 8, 16, 32] {
            let reduce = stride(from: width/2, through: 1, by: -1).filter { $0.nonzeroBitCount == 1 }
                .map { "sum += simd_shuffle_xor(sum, ushort(\($0)));" }.joined(separator: "\n")
            let kernel = original
                .replacingOccurrences(of: "uint row = tid / 32;", with: "uint row = tid / \(width); lane %= \(width);")
                .replacingOccurrences(of: "i += 32)", with: "i += \(width))")
                .replacingOccurrences(of: "float sum = simd_sum((accum.x + accum.y) + (accum.z + accum.w));", with: "float sum = (accum.x + accum.y) + (accum.z + accum.w);\n" + reduce)
            let modified = source.replacingCharacters(in: range, with: kernel)
            let options = MTLCompileOptions(); options.fastMathEnabled = false
            let library = try device.makeLibrary(source: modified, options: options)
            pipelines[width] = try device.makeComputePipelineState(function: library.makeFunction(name:"linear",constantValues:MTLFunctionConstantValues()))
        }
        for (name, shape) in [("lm_head.weight", [107520, 1536]),
                              ("model.layers.layers.0.mlp.gate_up_proj.weight", [12288, 1536]),
                              ("model.layers.layers.0.mlp.down_proj.weight", [1536, 6144])] {
            let w = try weights.tensor(name, shape)
            let input = device.makeBuffer(length: shape[1]*4)!
            let output = device.makeBuffer(length: shape[0]*4)!
            let pointer = input.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<shape[1] { pointer[i] = Float(i%17-8)/16 }
            var timings = [Int: [Double]]()
            var expected: [Float]?
            for round in 0..<6 {
                for width in (round%2 == 0 ? [4, 8, 16, 32] : [32, 16, 8, 4]) {
                    let command = queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
                    encoder.setComputePipelineState(pipelines[width]!)
                    encoder.setBuffer(w.buffer, offset: w.offset, index: 0)
                    encoder.setBuffer(w.buffer, offset: w.packing?.offsets ?? w.offset, index: 4)
                    encoder.setBuffer(input, offset: 0, index: 1); encoder.setBuffer(output, offset: 0, index: 2)
                    var a = Args(n: UInt32(shape[1]), m: UInt32(shape[0]), capacity: packed ? 1 : 0)
                    encoder.setBytes(&a, length: MemoryLayout<Args>.size, index: 3)
                    for _ in 0..<10 {
                        encoder.dispatchThreadgroups(MTLSize(width: (shape[0]*width+255)/256, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    }
                    encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                    XCTAssertEqual(command.status, .completed)
                    let actual = Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: Float.self), count: shape[0]))
                    if let expected = expected {
                        let error = zip(expected, actual).map { abs($0-$1) }.max()!
                        XCTAssertLessThanOrEqual(error, 0.125)
                    } else { expected = actual }
                    if round > 0 { timings[width, default: []].append((command.gpuEndTime-command.gpuStartTime)/10) }
                }
            }
            for width in [4, 8, 16, 32] {
                let values = timings[width]!.sorted()
                print("GEMV \(name) width=\(width) seconds=\(values[values.count/2])")
            }
        }
    }
}
