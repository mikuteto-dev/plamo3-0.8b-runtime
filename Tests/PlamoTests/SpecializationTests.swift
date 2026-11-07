import XCTest
import Metal
@testable import Plamo

final class SpecializationTests: XCTestCase {
    func testModelCosts() throws {
        guard ProcessInfo.processInfo.environment["PLAMO_SPECIALIZE_MODEL_TUNE"] == "1" else { throw XCTSkip("Set PLAMO_SPECIALIZE_MODEL_TUNE=1") }
        for packed in [false,true] {
            try autoreleasepool {
                let model = try Runtime(directory:"models/plamo",context:512,packed:packed)
                _ = try model.prefill([1]+(0..<325).map { 500+$0%37 })
                var samples = [[String:Any]]()
                for index in 0..<64 {
                    model.specializeWeights = index%4 == 1 || index%4 == 2
                    let gpu = model.gpuSeconds, start = Date.timeIntervalSinceReferenceDate
                    try model.forward([500+index%37])
                    samples.append(["specialized":model.specializeWeights,"gpu_seconds":model.gpuSeconds-gpu,
                                    "wall_seconds":Date.timeIntervalSinceReferenceDate-start])
                }
                XCTAssertEqual(model.position,390)
                XCTAssertTrue(model.logitValues().allSatisfy(\.isFinite))
                try json(["samples":samples,"metal_allocated_bytes":model.device.currentAllocatedSize,
                          "peak_metal_allocated_bytes":model.peakMetalBytes]).write(to:URL(fileURLWithPath:"results/weight-specialization-\(packed ? "packed" : "raw").json"),options:.atomic)
            }
        }
    }

    func testPrefillCosts() throws {
        guard ProcessInfo.processInfo.environment["PLAMO_SPECIALIZE_PREFILL_TUNE"] == "1" else { throw XCTSkip("Set PLAMO_SPECIALIZE_PREFILL_TUNE=1") }
        for packed in [false,true] {
            try autoreleasepool {
                let model = try Runtime(directory:"models/plamo",context:2048,packed:packed)
                let ids = Array(500..<516)
                for _ in 0..<64 { try model.forward(ids,output:false) }
                model.reset()
                var samples = [[String:Any]]()
                for index in 0..<64 {
                    model.specializeWeights = index%4 == 1 || index%4 == 2
                    let gpu = model.gpuSeconds, start = Date.timeIntervalSinceReferenceDate
                    try model.forward(ids,output:false)
                    samples.append(["specialized":model.specializeWeights,"gpu_seconds":model.gpuSeconds-gpu,
                                    "wall_seconds":Date.timeIntervalSinceReferenceDate-start])
                }
                XCTAssertEqual(model.position,1024)
                XCTAssertTrue(UnsafeBufferPointer(start:model.x.buffer.contents().assumingMemoryBound(to:Float.self),count:16*1536).allSatisfy(\.isFinite))
                try json(["samples":samples]).write(to:URL(fileURLWithPath:"results/weight-prefill-\(packed ? "packed" : "raw").json"),options:.atomic)
            }
        }
    }

    func testPackedKernelsMatchDynamic() throws { try checkKernels(packed:true) }

    func testRawMatrixMatchesDynamic() throws { try checkKernels(packed:false) }

    private func checkKernels(packed: Bool) throws {
        guard FileManager.default.fileExists(atPath:"models/plamo/model.safetensors") else { throw XCTSkip("Download weights first") }
        if packed && !FileManager.default.fileExists(atPath:"models/plamo/model.p12") { throw XCTSkip("Pack weights first") }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw XCTSkip("Metal unavailable") }
        let weights = try Weights(path:"models/plamo/model.safetensors",device:device,packedPath:packed ? "models/plamo/model.p12" : nil)
        let source = try String(contentsOf:Bundle.module.url(forResource:"Kernels",withExtension:"metal")!)
        let options = MTLCompileOptions(); options.fastMathEnabled = false
        let library = try device.makeLibrary(source:source,options:options)
        var pipelines = [[String:MTLComputePipelineState]]()
        for specialized in [false,true] {
            let constants = MTLFunctionConstantValues()
            if specialized {
                var enabled = packed
                constants.setConstantValue(&enabled,type:.bool,index:0)
            }
            var functions = [String:MTLComputePipelineState]()
            for name in (packed ? ["linear","linear_pair","matrix"] : ["matrix"]) {
                functions[name] = try device.makeComputePipelineState(function:library.makeFunction(name:name,constantValues:constants))
            }
            pipelines.append(functions)
        }
        let tune = ProcessInfo.processInfo.environment["PLAMO_SPECIALIZE_TUNE"] == "1"
        for (name,n,m) in [("lm_head.weight",1536,107520),("model.layers.layers.0.mixer.o_proj.weight",2048,1536),("model.layers.layers.0.mlp.gate_up_proj.weight",1536,12288),("model.layers.layers.0.mlp.down_proj.weight",6144,1536)] {
            let w = try weights.tensor(name,[m,n])
            let cases = packed ? [("linear",1),("linear_pair",2),("matrix",2),("matrix",15),("matrix",16)]
                               : [("matrix",2),("matrix",15),("matrix",16)]
            for (kernel,batch) in cases {
                let input = device.makeBuffer(length:batch*n*4)!, output = device.makeBuffer(length:batch*m*4)!
                let values = input.contents().assumingMemoryBound(to:Float.self)
                for i in 0..<batch*n { values[i] = Float(i%17-8)/16 }
                var expected = [UInt32](), times = [Int:[Double]]()
                for variant in (tune ? [0,1,1,0,0,1] : [0,1]) {
                    let command = queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
                    encoder.setComputePipelineState(pipelines[variant][kernel]!)
                    encoder.setBuffer(w.buffer,offset:w.offset,index:0); encoder.setBuffer(input,offset:0,index:1)
                    encoder.setBuffer(output,offset:0,index:2)
                    encoder.setBuffer(w.buffer,offset:w.packing?.offsets ?? w.offset,index:4)
                    var a = Args(n:UInt32(n),m:UInt32(m),batch:UInt32(batch),capacity:packed ? 1 : 0)
                    encoder.setBytes(&a,length:MemoryLayout<Args>.size,index:3)
                    for _ in 0..<(tune ? 8 : 1) {
                        encoder.dispatchThreadgroups(MTLSize(width:kernel == "matrix" ? (m+31)/32 : (m*32+255)/256,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1))
                    }
                    encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                    XCTAssertEqual(command.status,.completed)
                    let actual = Array(UnsafeBufferPointer(start:output.contents().assumingMemoryBound(to:UInt32.self),count:batch*m))
                    if expected.isEmpty { expected = actual }
                    XCTAssertTrue(actual.elementsEqual(expected),"\(name), batch=\(batch)")
                    times[variant,default:[]].append((command.gpuEndTime-command.gpuStartTime)/Double(tune ? 8 : 1))
                }
                if tune {
                    for variant in [0,1] {
                        let samples = times[variant]!.sorted()
                        print("WEIGHT_CONST \(name) packed=\(packed) batch=\(batch) specialized=\(variant) ms=\(samples[samples.count/2]*1000)")
                    }
                }
            }
        }
    }
}
