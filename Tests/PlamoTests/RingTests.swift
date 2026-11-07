import XCTest
import Metal
@testable import Plamo

final class RingTests: XCTestCase {
    func testHoistedModulo() throws {
        let tune=ProcessInfo.processInfo.environment["PLAMO_RING_TUNE"] == "1"
        let device=try XCTUnwrap(MTLCreateSystemDefaultDevice()),queue=try XCTUnwrap(device.makeCommandQueue())
        let source=try String(contentsOf:Bundle.module.url(forResource:"Kernels",withExtension:"metal")!)
        let begin=try XCTUnwrap(source.range(of:"kernel void attend(")).lowerBound
        let end=try XCTUnwrap(source.range(of:"// Keep each 256-key partition")).lowerBound
        var candidate=String(source[begin..<end]).replacingOccurrences(of:"kernel void attend(",with:"kernel void attend_modulo(")
        let normalization="    valueBase %= a.capacity;\n"
        XCTAssertEqual(candidate.components(separatedBy:normalization).count,2)
        candidate=candidate.replacingOccurrences(of:normalization,with:"")
        let index="(physical >= a.capacity ? physical-a.capacity : physical)*128+row+d"
        XCTAssertEqual(candidate.components(separatedBy:index).count,2)
        candidate=candidate.replacingOccurrences(of:index,with:"((valueBase+key)%a.capacity)*128+row+d")
        let options=MTLCompileOptions();options.fastMathEnabled=false
        let library=try device.makeLibrary(source:source+"\n"+candidate,options:options)
        var pipelines=[String:MTLComputePipelineState]()
        for name in ["attend","attend_modulo"] { pipelines[name]=try device.makeComputePipelineState(function:try XCTUnwrap(library.makeFunction(name:name))) }
        var records=[[String:Any]]()
        for (count,capacity,batch,base,mode) in [(1,1,1,0,0),(19,19,3,0,0),(65,79,1,236,1),(2048,2064,1,4111,1),(2063,2064,16,2049,1),(2048,2064,2,2043,2),(131,131,3,0,3),(32768,32768,1,0,0),(32768,32768,16,0,0)] {
            autoreleasepool {
                XCTAssertLessThanOrEqual(count,capacity)
                let rows=mode == 3 ? batch*count : capacity
                let values=(0..<rows*128).map { UInt16((Float(($0*13)%41-20)/32).bitPattern>>16) }
                let scores=(0..<batch*16*count).map { Float(($0*7)%23+1)/Float(count*12) }
                let v=values.withUnsafeBytes { device.makeBuffer(bytes:$0.baseAddress!,length:$0.count)! }
                let s=scores.withUnsafeBytes { device.makeBuffer(bytes:$0.baseAddress!,length:$0.count)! }
                let output=device.makeBuffer(length:batch*2048*4)!
                var expected=[UInt32](),times=[Bool:[Double]](),equal=true
                for round in 0..<(tune ? 12 : 2) {
                    let optimized=tune ? [false,true,true,false][round%4] : round == 1
                    let command=queue.makeCommandBuffer()!,encoder=command.makeComputeCommandEncoder()!
                    encoder.setComputePipelineState(pipelines[optimized ? "attend" : "attend_modulo"]!)
                    encoder.setBuffer(s,offset:0,index:0);encoder.setBuffer(v,offset:0,index:1);encoder.setBuffer(output,offset:0,index:2)
                    var a=Args(n:mode == 3 ? 1 : 0,batch:UInt32(batch),start:mode == 2 ? 4090 : UInt32(base+count-batch),count:UInt32(count),capacity:UInt32(capacity),base:UInt32(base),full:mode == 0 || mode == 3 ? 1 : 0,offset:mode == 2 ? 1 : 0)
                    encoder.setBytes(&a,length:MemoryLayout<Args>.size,index:3)
                    for _ in 0..<(tune ? 8 : 1) { encoder.dispatchThreadgroups(MTLSize(width:4,height:batch,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1)) }
                    encoder.endEncoding();command.commit();command.waitUntilCompleted()
                    XCTAssertEqual(command.status,.completed,"\(String(describing:command.error))")
                    let actual=Array(UnsafeBufferPointer(start:output.contents().assumingMemoryBound(to:UInt32.self),count:batch*2048))
                    if expected.isEmpty { expected=actual }
                    equal=equal && actual==expected
                    XCTAssertEqual(actual,expected,"count=\(count) capacity=\(capacity) batch=\(batch) base=\(base) mode=\(mode)")
                    if round>=4 { times[optimized,default:[]].append((command.gpuEndTime-command.gpuStartTime)/8) }
                }
                if tune {
                    var medians=[String:Double]()
                    for optimized in [false,true] {
                        let t=times[optimized]!.sorted();medians[optimized ? "hoisted" : "modulo"]=(t[1]+t[2])/2
                    }
                    records.append(["count":count,"capacity":capacity,"batch":batch,"base":base,"mode":mode,"bitwise_equal":equal,"median_seconds":medians,"samples":["modulo":times[false]!,"hoisted":times[true]!]])
                    print("RING count=\(count) batch=\(batch) mode=\(mode) medians=\(medians)")
                }
            }
        }
        if tune { try json(records).write(to:URL(fileURLWithPath:"results/ring-attend.json"),options:.atomic) }
    }
}
