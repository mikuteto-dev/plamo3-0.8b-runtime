import XCTest
import Metal
@testable import Plamo

final class BatchMapTests: XCTestCase {
    func testUnusedKVSurvivesPrefillAndNeverAffectsShorterInputs() throws {
        guard ProcessInfo.processInfo.environment["PLAMO_BATCH_MAPS"] == "1" else { throw XCTSkip("Set PLAMO_BATCH_MAPS=1") }
        let ring = ProcessInfo.processInfo.environment["PLAMO_BATCH_CLEAR_RING"] == "1"
        let context = ring ? 2100 : 128, steps = ring ? 2 : 1
        let layouts = ring ? [[2063,3],[3,4]] : [[63,3,7,5],[3,63],[1,4,2],[2]]
        var records = [[String:Any]]()
        for packed in [false,true] {
            try autoreleasepool {
                let scalar = try Runtime(directory:"models/plamo",context:context,packed:packed), batch = try Batch(scalar)
                // NaN-filled unused KV must never contribute to attention.
                for buffers in batch.kv {
                    for ref in [buffers.keys,buffers.values] { memset(ref.buffer.contents()+ref.offset,0xff,ref.buffer.length-ref.offset) }
                }
                for lengths in layouts {
                    let prompts = lengths.map { n in [1]+(1..<n).map { 1000+$0*17 } }
                    func state(_ lane: Int, scalarOnly: Bool = false) -> [Data] {
                        let row = scalarOnly ? 0 : lane
                        var result = [(scalarOnly ? scalar.normalized : batch.hidden,1536*4),
                                      (scalarOnly ? scalar.logits : batch.logits,107520*2)].map { ref,bytes in
                            Data(bytes:ref.buffer.contents()+ref.offset+row*bytes,count:bytes)
                        }
                        for (i,layer) in scalar.layers.enumerated() {
                            for ref in scalarOnly ? [layer.keys,layer.values] : [batch.kv[i].keys,batch.kv[i].values] {
                                result.append(Data(bytes:ref.buffer.contents()+ref.offset+row*layer.capacity*128*2,count:min(lengths[lane]+steps,layer.capacity)*128*2))
                            }
                        }
                        return result
                    }
                    func tails() -> [Data] {
                        (0..<4).flatMap { lane in
                            scalar.layers.enumerated().flatMap { i,layer in
                                let used = lane < lengths.count ? min(lengths[lane],layer.capacity) : 0
                                return [batch.kv[i].keys,batch.kv[i].values].map { ref in
                                    Data(bytes:ref.buffer.contents()+ref.offset+(lane*layer.capacity+used)*128*2,count:(layer.capacity-used)*128*2)
                                }
                            }
                        }
                    }
                    var tokens = [Int](), expected = [[Data]](), next = [Int]()
                    for lane in prompts.indices {
                        scalar.reset()
                        let token = try scalar.prefill(prompts[lane]); tokens.append(token)
                        var prediction = token
                        for _ in 0..<steps { prediction = try scalar.forward([prediction]) }
                        next.append(prediction); expected.append(state(lane,scalarOnly:true))
                    }
                    let before = tails()
                    _ = try batch.prefill(prompts)
                    let untouched = tails().elementsEqual(before)
                    XCTAssertTrue(untouched,"Unused KV overwritten: packed=\(packed) lengths=\(lengths)")
                    var actual = tokens
                    for _ in 0..<steps { actual = try batch.step(actual.map { Optional($0) }) }
                    var equal = true
                    for lane in prompts.indices {
                        equal = state(lane).elementsEqual(expected[lane]) && equal
                        let values = batch.logits.buffer.contents().advanced(by:batch.logits.offset+lane*107520*2).assumingMemoryBound(to:UInt16.self)
                        XCTAssertTrue(UnsafeBufferPointer(start:values,count:107520).allSatisfy { ($0 & 0x7f80) != 0x7f80 })
                    }
                    XCTAssertTrue(equal,"State mismatch: packed=\(packed) lengths=\(lengths)")
                    XCTAssertEqual(actual,next)
                    records.append(["packed":packed,"lengths":lengths,"steps":steps,"unused_kv_untouched":untouched,"state_bits_equal":equal,"predictions_equal":actual==next])
                }
            }
        }
        let validation = ProcessInfo.processInfo.environment["MTL_SHADER_VALIDATION"] == "1"
        try json(["cases":records,"shader_validation":validation,"context":context] as [String:Any])
            .write(to:URL(fileURLWithPath:"results/batch-lazy\(ring ? "-ring" : "")-\(validation ? "validation" : "correctness").json"),options:.atomic)
    }

    func testCachedIndicesMatchRebuildingAcrossCompactionAndRingWraps() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        for (stride,capacity) in [(1,1),(8,8),(2100,2064),(8191,8191)] {
            let size = 4*stride+4
            func allocate(_ bytes: Int) throws -> Ref {
                Ref(buffer:try XCTUnwrap(device.makeBuffer(length:bytes,options:.storageModeShared)))
            }
            let indices = try allocate(size*4)
            memset(indices.buffer.contents(),0xff,size*4)
            let map = try Batch.Geometry(indices:indices,counts:allocate(16),slots:allocate(16))
            var expected = [UInt32](repeating:UInt32.max,count:size)
            let count = min(capacity,2048)
            let events = [(0,3,0,1),(0,3,0,count),(0,3,0,1),(0,3,0,count),
                          (1,2,capacity-1,count),(0,1,capacity-2 < 0 ? 0 : capacity-2,count),
                          (0,3,capacity+1,count),(1,0,capacity,count),(1,0,2*capacity,count),
                          (0,3,0,1),(0,3,0,count)]
            for (query,lane,first,length) in events {
                map.update(query,lane:lane,first:first,count:length,stride:stride,capacity:capacity)
                for key in 0..<length { expected[query*stride+key] = UInt32(lane*capacity+(first+key)%capacity) }
                let actual = Array(UnsafeBufferPointer(start:indices.buffer.contents().assumingMemoryBound(to:UInt32.self),count:size))
                XCTAssertEqual(actual,expected,"stride=\(stride) query=\(query) lane=\(lane) first=\(first)")
            }
        }
    }

    func testGenerationWithoutClearingCost() throws {
        guard ProcessInfo.processInfo.environment["PLAMO_BATCH_CLEAR_COST"] == "1" else { throw XCTSkip("Set PLAMO_BATCH_CLEAR_COST=1") }
        XCTAssertNotEqual(ProcessInfo.processInfo.environment["MTL_SHADER_VALIDATION"],"1")
        let tokenizer = try Tokenizer(path:"models/plamo/tokenizer.jsonl")
        let texts = ["The meeting has been moved to next Monday.","Please send the updated report before Friday.",
                     "The library will be closed during the holidays.","We will notify you when the package has arrived."]
        let prompts = try texts.map { tokenizer.encode(try Tokenizer.prompt($0,target:"ja"),bos:true) }
        let batch = try Batch(Runtime(directory:"models/plamo",context:2304,packed:true))
        let bytes = batch.kv.reduce(0) { $0+$1.keys.buffer.length+$1.values.buffer.length }
        var expected: [[Int]]?, samples = [[String:Any]]()
        for trial in -1..<4 {
            for clear in trial%2 == 0 ? [true,false] : [false,true] {
                let start = DispatchTime.now().uptimeNanoseconds
                if clear {
                    for buffers in batch.kv {
                        for ref in [buffers.keys,buffers.values] { memset(ref.buffer.contents(),0xcd,ref.buffer.length) }
                    }
                }
                let prepared = DispatchTime.now().uptimeNanoseconds
                let output = try batch.generate(prompts,limits:[128,128,128,128])
                let end = DispatchTime.now().uptimeNanoseconds
                let ids = output.map(\.ids)
                if expected == nil { expected = ids }
                let equal = ids == expected!
                XCTAssertTrue(equal);XCTAssertTrue(output.allSatisfy { $0.reason == "stop" })
                samples.append(["trial":trial,"warmup":trial<0,"clear":clear,"output_ids":ids,"ids_equal":equal,
                                "all_eos":output.allSatisfy { $0.reason == "stop" },"cleared_bytes":clear ? bytes : 0,
                                "clear_seconds":clear ? Double(prepared-start)/1e9 : 0,
                                "wall_seconds":Double(end-start)/1e9,"gpu_seconds":batch.gpuSeconds])
            }
        }
        try json(["samples":samples,"context":2304,"packed":true,"kv_bytes":bytes,
                  "scope":"Same runtime, weights and buffers. Four short real prompts; one warm-up pair and four alternating-order pairs. Baseline emulates the removed prefill memset immediately before generation. Setup, tokenization, CLI, text decoding and HTTP are not timed."] as [String:Any])
            .write(to:URL(fileURLWithPath:"results/batch-lazy-warm.json"),options:.atomic)
    }

    func testIndexUpdateCost() throws {
        guard ProcessInfo.processInfo.environment["PLAMO_BATCH_MAP_COST"] == "1" else { throw XCTSkip("Set PLAMO_BATCH_MAP_COST=1") }
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        var records = [[String:Any]]()
        for context in [695,2100,8191] {
            func allocate() throws -> Ref {
                let ref = Ref(buffer:try XCTUnwrap(device.makeBuffer(length:(4*context+4)*4,options:.storageModeShared)))
                memset(ref.buffer.contents(),0xff,ref.buffer.length)
                return ref
            }
            let original = try [allocate(),allocate()]
            let maps = try (0..<2).map { _ in try Batch.Geometry(indices:allocate(),counts:allocate(),slots:allocate()) }
            let begin = context/2, steps = min(128,context-begin)
            func fill(_ cached: Bool) {
                for _ in 0..<24 {
                    for position in begin..<begin+steps {
                        for full in 0..<2 {
                            let capacity = full == 1 ? context : min(context,2064)
                            let first = full == 1 ? 0 : max(0,position-2047), count = position-first+1
                            for query in 0..<4 {
                                if cached { maps[full].update(query,lane:query,first:first,count:count,stride:context,capacity:capacity) }
                                else {
                                    let target = original[full].buffer.contents().assumingMemoryBound(to:UInt32.self)+query*context
                                    for key in 0..<count { target[key] = UInt32(query*capacity+(first+key)%capacity) }
                                }
                            }
                        }
                    }
                }
            }
            fill(false); fill(true)
            var samples = [[String:Any]]()
            for trial in 0..<4 {
                for cached in trial%2 == 0 ? [false,true] : [true,false] {
                    let start = DispatchTime.now().uptimeNanoseconds
                    fill(cached)
                    let seconds = Double(DispatchTime.now().uptimeNanoseconds-start)/1e9
                    samples.append(["cached":cached,"seconds":seconds,"trial":trial])
                }
            }
            for i in 0..<2 {
                XCTAssertEqual(Data(bytes:original[i].buffer.contents(),count:(4*context+4)*4),
                               Data(bytes:maps[i].indices.buffer.contents(),count:(4*context+4)*4))
            }
            records.append(["context":context,"steps_per_pass":steps,"passes":24,"samples":samples])
        }
        try json(["cases":records,"scope":"CPU index updates only, stable four active lanes; one warm-up and four alternating-order pairs. Not translation latency or GPU throughput."])
            .write(to:URL(fileURLWithPath:"results/batch-map-cost.json"),options:.atomic)
    }

    func testCompactedAttentionMatchesScalarForBothWeightFormats() throws {
        guard ProcessInfo.processInfo.environment["PLAMO_BATCH_MAPS"] == "1" else { throw XCTSkip("Set PLAMO_BATCH_MAPS=1") }
        for (packed,context) in [(false,24),(true,24),(false,8191),(true,8191)] {
            try autoreleasepool {
                let scalar = try Runtime(directory:"models/plamo",context:context,packed:packed), batch = try Batch(scalar)
                let prompts = [2,3,5,8].map { n in [1]+(1..<n).map { 1000+$0*17 } }
                func state(_ lane: Int, scalarOnly: Bool = false) -> [Data] {
                    let row = scalarOnly ? 0 : lane
                    var result = [(scalarOnly ? scalar.normalized : batch.hidden,1536*4),
                                  (scalarOnly ? scalar.logits : batch.logits,107520*2)].map { ref,bytes in
                        Data(bytes:ref.buffer.contents()+ref.offset+row*bytes,count:bytes)
                    }
                    for (i,layer) in scalar.layers.enumerated() {
                        for ref in scalarOnly ? [layer.keys,layer.values] : [batch.kv[i].keys,batch.kv[i].values] {
                            result.append(Data(bytes:ref.buffer.contents()+ref.offset+row*layer.capacity*128*2,
                                               count:(prompts[lane].count+1)*128*2))
                        }
                    }
                    return result
                }
                var tokens = [Int](), expected = [[Data]](), next = [Int]()
                for lane in prompts.indices {
                    scalar.reset()
                    let token = try scalar.prefill(prompts[lane]); tokens.append(token)
                    next.append(try scalar.forward([token])); expected.append(state(lane,scalarOnly:true))
                }
                for mask in 1..<16 {
                    let lanes = prompts.indices.filter { mask & (1 << $0) != 0 }
                    _ = try batch.prefill(prompts)
                    let frozen = prompts.indices.map { state($0) }
                    let actual = try batch.step(prompts.indices.map { lanes.contains($0) ? tokens[$0] : nil })
                    for lane in prompts.indices {
                        XCTAssertEqual(state(lane),lanes.contains(lane) ? expected[lane] : frozen[lane],"packed=\(packed) lanes=\(lanes) lane=\(lane)")
                        if lanes.contains(lane) { XCTAssertEqual(actual[lane],next[lane]) }
                    }
                }
            }
        }
    }
}
