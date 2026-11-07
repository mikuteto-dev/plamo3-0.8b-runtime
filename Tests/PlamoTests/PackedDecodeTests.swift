import XCTest
import Metal
@testable import Plamo

final class PackedDecodeTests: XCTestCase {
    func testPackedFourPreservesAllBitPatterns() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw XCTSkip("Metal unavailable") }
        var payload = [UInt8](), offsets = [UInt32](), expected = [UInt32]()
        for block in 0..<512 {
            offsets.append(UInt32(payload.count))
            for i in 0..<128 {
                let bits = block*128+i
                payload.append(UInt8(truncatingIfNeeded:bits)); payload.append(UInt8(bits>>8))
                expected.append(UInt32(bits)<<16)
            }
        }
        for minimum in 108...170 {
            for block in 0..<32 {
                offsets.append(UInt32(payload.count)|UInt32(minimum-107))
                var bytes = [UInt8](repeating:0,count:192)
                for i in 0..<128 {
                    let index = block*128+i, sm = index/16, code = index%16
                    bytes[i] = UInt8(sm)
                    bytes[128+i/2] |= UInt8(code<<((i%2)*4))
                    let exponent = code == 15 ? 0 : minimum+code
                    expected.append(UInt32((sm&127)|((sm&128)<<8)|(exponent<<7))<<16)
                }
                payload.append(contentsOf:bytes)
            }
        }
        XCTAssertEqual(expected.count,323_584)
        let source = try String(contentsOf:Bundle.module.url(forResource:"Kernels",withExtension:"metal")!) + """

        kernel void check_packed(device const ushort* w [[buffer(0)]],
                                 device const uint* offsets [[buffer(1)]],
                                 device uint4* output [[buffer(2)]],
                                 device uint4* scalar [[buffer(3)]],
                                 uint i [[thread_position_in_grid]]) {
            output[i] = as_type<uint4>(packed_four(w,offsets,i*4));
            scalar[i] = uint4(packed_bits(w,offsets,i*4),packed_bits(w,offsets,i*4+1),
                              packed_bits(w,offsets,i*4+2),packed_bits(w,offsets,i*4+3)) << 16;
        }
        """
        let options = MTLCompileOptions(); options.fastMathEnabled = false
        let library = try device.makeLibrary(source:source,options:options)
        let pipeline = try device.makeComputePipelineState(function:try XCTUnwrap(library.makeFunction(name:"check_packed")))
        let data = payload.withUnsafeBytes { device.makeBuffer(bytes:$0.baseAddress!,length:$0.count)! }
        let table = offsets.withUnsafeBytes { device.makeBuffer(bytes:$0.baseAddress!,length:$0.count)! }
        let output = device.makeBuffer(length:expected.count*4)!, scalar = device.makeBuffer(length:expected.count*4)!
        let command = queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        for (i,buffer) in [data,table,output,scalar].enumerated() { encoder.setBuffer(buffer,offset:0,index:i) }
        encoder.dispatchThreadgroups(MTLSize(width:expected.count/4/256,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        XCTAssertEqual(command.status,.completed)
        for buffer in [output,scalar] {
            let actual = UnsafeBufferPointer(start:buffer.contents().assumingMemoryBound(to:UInt32.self),count:expected.count)
            XCTAssertTrue(actual.elementsEqual(expected))
        }
    }
}
