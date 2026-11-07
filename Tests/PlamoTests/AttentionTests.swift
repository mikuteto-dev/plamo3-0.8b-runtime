import XCTest
import Metal
@testable import Plamo

final class AttentionTests: XCTestCase {
    func testPartitionedAttention() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw XCTSkip("Metal unavailable") }
        let source = try String(contentsOf: Bundle.module.url(forResource: "Kernels", withExtension: "metal")!)
        let options = MTLCompileOptions(); options.fastMathEnabled = false
        let library = try device.makeLibrary(source: source, options: options)
        var pipelines = [String: MTLComputePipelineState]()
        for name in ["matrix","softmax","attend","attention","attention_merge"] {
            pipelines[name] = try device.makeComputePipelineState(function: library.makeFunction(name:name,constantValues:MTLFunctionConstantValues()))
        }
        let tune = ProcessInfo.processInfo.environment["PLAMO_ATTENTION_TUNE"] == "1"
        for (count,batch,mode) in [(19,3,0),(513,16,0),(2048,3,1),(2050,3,2),(131,3,3),(4096,16,0),(8192,16,0),(32768,1,0),(32768,16,0)] {
            let full = mode == 0 || mode == 3
            let separate = mode == 1 || mode == 3
            let capacity = full ? count : 2064
            let start = full ? count-batch : 4090
            let base = full || mode == 1 ? 0 : start-2047
            let keyCount = (separate ? batch : 1)*count*128
            let keys = (0..<keyCount).map { UInt16((Float(($0*17)%43-21)/32).bitPattern >> 16) }
            let values = (0..<(mode == 3 ? batch*count : capacity)*128).map { UInt16((Float(($0*13)%41-20)/32).bitPattern >> 16) }
            let queries = (0..<batch*2048).map { Float(($0*7)%47-23)/32 }
            let counts = (0..<batch).map { UInt32(count-batch+$0+1) }
            let k = keys.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!,length: $0.count)! }
            let v = values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!,length: $0.count)! }
            let q = queries.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!,length: $0.count)! }
            let c = counts.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!,length: $0.count)! }
            let scores = device.makeBuffer(length: batch*16*count*4)!
            let partitions = (count+255)/256
            let partBytes = batch*partitions*16*128*4
            let scratch = device.makeBuffer(length: partBytes+batch*partitions*16*8)!
            let output = device.makeBuffer(length: batch*2048*4)!
            let a = Args(n: mode == 3 ? 1 : 0,m: UInt32(partitions),batch: UInt32(batch),start: UInt32(start),
                         count: UInt32(count),capacity: UInt32(capacity),base: UInt32(base),full: full ? 1 : 0,offset: mode == 1 ? 1 : 0)
            var expected = [Float](), timings = [Bool:[Double]]()
            for fused in (tune ? [false,true,true,false,false,true] : [false,true]) {
                let command = queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
                func run(_ name: String, _ bindings: [(Int,MTLBuffer,Int)], _ args: Args, _ width: Int, _ height: Int, _ threads: Int) {
                    encoder.setComputePipelineState(pipelines[name]!)
                    for (i,b,offset) in bindings { encoder.setBuffer(b,offset: offset,index: i) }
                    var args = args
                    encoder.setBytes(&args,length: MemoryLayout<Args>.size,index: 3)
                    encoder.dispatchThreadgroups(MTLSize(width: width,height: height,depth: 1),threadsPerThreadgroup: MTLSize(width: threads,height: 1,depth: 1))
                }
                for _ in 0..<(tune ? 8 : 1) {
                    if fused {
                        run("attention",[(0,k,0),(1,q,0),(2,partitions == 1 ? output : scratch,0),(4,v,0),(5,scratch,partBytes),(6,c,0)],a,partitions,batch,256)
                        if partitions > 1 { run("attention_merge",[(0,scratch,0),(1,scratch,partBytes),(2,output,0)],a,batch*16,1,128) }
                    } else {
                        run("matrix",[(0,k,0),(1,q,0),(2,scores,0),(4,k,0)],
                            Args(n: 128,m: UInt32(count),batch: UInt32(batch*16),start: separate ? 1 : 0,full: 1),(count+31)/32,batch,256)
                        run("softmax",[(0,scores,0),(1,c,0)],a,batch*16,1,256)
                        run("attend",[(0,scores,0),(1,v,0),(2,output,0)],a,4,batch,256)
                    }
                }
                encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                XCTAssertEqual(command.status,.completed,"\(String(describing: command.error))")
                let actual = Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: Float.self),count: batch*2048))
                if expected.isEmpty { expected = actual }
                let error = zip(actual,expected).map { abs($0-$1) }.max()!
                XCTAssertLessThanOrEqual(error,0.002,"count=\(count), batch=\(batch), mode=\(mode)")
                XCTAssertTrue(actual.allSatisfy(\.isFinite))
                timings[fused,default:[]].append((command.gpuEndTime-command.gpuStartTime)/Double(tune ? 8 : 1))
            }
            if tune {
                for fused in [false,true] {
                    let values = timings[fused]!.sorted()
                    print("ATTENTION count=\(count) batch=\(batch) mode=\(mode) fused=\(fused) ms=\(values[values.count/2]*1000)")
                }
            }
        }
    }
}
