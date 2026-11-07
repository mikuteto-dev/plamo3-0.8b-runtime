import XCTest
import Metal
@testable import Plamo

final class QSATests: XCTestCase {
    func testSelectionCausalityTailsAndTies() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw XCTSkip("Metal unavailable") }
        let source = try String(contentsOf: Bundle.module.url(forResource: "Kernels", withExtension: "metal")!)
        let options = MTLCompileOptions(); options.fastMathEnabled = false
        let library = try device.makeLibrary(source: source, options: options)
        let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "qsa_select")!)
        for visible in [1,4,7,19,64,2051,16387] {
            for budget in [4,16,2048] {
                let batch = 3, blocks = (visible+batch-1)/4
                var scores = [Float](repeating: 0, count: batch*max(1,blocks))
                for t in 0..<batch {
                    for b in 0..<blocks { scores[t*blocks+b] = b < (visible+t)/4 ? Float((b*17+t*3)%23)/8 : 1e9 }
                }
                let input = scores.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)! }
                let indices = device.makeBuffer(length: batch*(budget+3)*4)!
                let counts = device.makeBuffer(length: batch*4)!
                let command = queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(input, offset: 0, index: 0); encoder.setBuffer(indices, offset: 0, index: 1); encoder.setBuffer(counts, offset: 0, index: 2)
                var a = Args(m: UInt32(budget), batch: UInt32(batch), start: UInt32(visible-1), count: UInt32(blocks))
                encoder.setBytes(&a, length: MemoryLayout<Args>.size, index: 3)
                encoder.dispatchThreadgroups(MTLSize(width: batch, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                XCTAssertEqual(command.status, .completed)
                let selected = indices.contents().assumingMemoryBound(to: UInt32.self)
                let sizes = counts.contents().assumingMemoryBound(to: UInt32.self)
                for t in 0..<batch {
                    let complete = (visible+t)/4
                    let ranked = (0..<complete).sorted {
                        let a = scores[t*blocks+$0], b = scores[t*blocks+$1]
                        return a == b ? $0 < $1 : a > b
                    }
                    let picked = ranked.prefix(budget/4).sorted()
                    let expected = picked.flatMap { b in (0..<4).map { b*4+$0 } } + Array((complete*4)..<(visible+t))
                    XCTAssertEqual(Int(sizes[t]), expected.count)
                    let actual = (0..<Int(sizes[t])).map { Int(selected[t*(budget+3)+$0]) }
                    XCTAssertEqual(actual, expected, "visible=\(visible+t), budget=\(budget)")
                    XCTAssertEqual(Set(actual).count, actual.count)
                    XCTAssertTrue(actual.allSatisfy { $0 < visible+t })
                }
            }
        }
    }

    func testVisibleTailKeepsAllKeys() throws {
        guard FileManager.default.fileExists(atPath: "models/plamo/auxiliary.safetensors") else { throw XCTSkip("Train adapters first") }
        let dense = try Runtime(directory: "models/plamo", context: 19, packed: true)
        let sparse = try Runtime(directory: "models/plamo", context: 19, packed: true, qsaBudget: 16)
        let ids = [1]+Array(500..<517)
        XCTAssertEqual(try dense.prefill(ids), try sparse.prefill(ids))
        XCTAssertEqual(dense.logitValues().map(\.bitPattern), sparse.logitValues().map(\.bitPattern))
        XCTAssertGreaterThan(sparse.sparseQueries, 0)
        XCTAssertEqual(sparse.sparseKeys, sparse.denseKeys)
        XCTAssertEqual(sparse.layers[7].indexer!.rawKeys.buffer.length, 19*128*2)
    }

    func testDenseBudgetMatchesOriginal() throws {
        guard FileManager.default.fileExists(atPath: "models/plamo/auxiliary.safetensors") else { throw XCTSkip("Train adapters first") }
        let dense = try Runtime(directory: "models/plamo", context: 64, packed: true)
        let sparse = try Runtime(directory: "models/plamo", context: 64, packed: true, qsaBudget: 128)
        let ids = [1,500,600,700,800]
        XCTAssertEqual(try dense.prefill(ids), try sparse.prefill(ids))
        XCTAssertEqual(dense.logitValues().map(\.bitPattern), sparse.logitValues().map(\.bitPattern))
        XCTAssertEqual(sparse.sparseQueries, 0)
    }
}
