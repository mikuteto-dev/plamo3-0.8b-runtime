import XCTest
import Metal
@testable import Plamo

final class ArgmaxTests: XCTestCase {
    func testSelection() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw XCTSkip("Metal unavailable") }
        let source = try String(contentsOf:Bundle.module.url(forResource:"Kernels",withExtension:"metal")!)
        let options = MTLCompileOptions(); options.fastMathEnabled = false
        let library = try device.makeLibrary(source:source,options:options)
        let pipeline = try device.makeComputePipelineState(function:try XCTUnwrap(library.makeFunction(name:"argmax")))
        XCTAssertGreaterThanOrEqual(pipeline.maxTotalThreadsPerThreadgroup,512)
        for count in [1,31,32,33,255,256,257,1023,1024,1025,107520] {
            for batch in [1,2,3] {
                for pattern in 0..<5 {
                    var values = (0..<count*batch).map { Float(($0*17)%4093-2046)/32 }
                    for row in 0..<batch {
                        switch pattern {
                        case 1:
                            for i in 0..<count { values[row*count+i] = -.infinity }
                        case 2:
                            for i in 0..<count { values[row*count+i] = .nan }
                        case 3:
                            for i in 0..<count { values[row*count+i] = i%2 == 0 ? -0.0 : 0.0 }
                        case 4:
                            values[row*count] = .nan
                            if count > 1 { values[row*count+count-1] = .infinity }
                            if count > 257 { values[row*count+257] = .infinity }
                        default: break
                        }
                    }
                    for compact in [false,true] {
                        let bits = values.map { value -> UInt16 in
                            let u = value.bitPattern
                            return UInt16(truncatingIfNeeded:(u &+ 0x7fff &+ ((u>>16)&1))>>16)
                        }
                        let tested = compact ? bits.map { Float(bitPattern:UInt32($0)<<16) } : values
                        let expected = (0..<batch).map { row -> UInt32 in
                            var best = -Float.infinity, index = UInt32.max
                            for i in 0..<count {
                                let value = tested[row*count+i]
                                if value > best || (value == best && UInt32(i) < index) { best = value; index = UInt32(i) }
                            }
                            return index
                        }
                        let data = compact ? bits.withUnsafeBytes { Data($0) } : values.withUnsafeBytes { Data($0) }
                        let input = data.withUnsafeBytes { device.makeBuffer(bytes:$0.baseAddress!,length:$0.count)! }
                        let output = device.makeBuffer(length:batch*4)!
                        let command = queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
                        encoder.setComputePipelineState(pipeline)
                        encoder.setBuffer(input,offset:0,index:0); encoder.setBuffer(output,offset:0,index:1)
                        var args = Args(n:UInt32(count),full:compact ? 2 : 0)
                        encoder.setBytes(&args,length:MemoryLayout<Args>.size,index:3)
                        encoder.dispatchThreadgroups(MTLSize(width:batch,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:512,height:1,depth:1))
                        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                        XCTAssertEqual(command.status,.completed)
                        let actual = Array(UnsafeBufferPointer(start:output.contents().assumingMemoryBound(to:UInt32.self),count:batch))
                        XCTAssertEqual(actual,expected,"count=\(count) batch=\(batch) pattern=\(pattern) compact=\(compact)")
                    }
                }
            }
        }
    }
}
