import XCTest
import CryptoKit
@testable import Plamo

final class MaximumTests: XCTestCase {
    func testDecodeCosts() throws {
        guard ProcessInfo.processInfo.environment["PLAMO_DECODE_TUNE"] == "1" else { throw XCTSkip("Set PLAMO_DECODE_TUNE=1") }
        let tokens = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath:"results/reference-long-32768.tokens.json"))) as? [Int])
        for length in [8192,16384] {
            try autoreleasepool {
                let runtime = try Runtime(directory:"models/plamo",context:length+32,packed:true)
                _ = try runtime.prefill(Array(tokens.prefix(length)))
                var samples = [[String:Any]]()
                for (index,token) in tokens[length..<length+32].enumerated() {
                    runtime.splitAttention = index%4 == 1 || index%4 == 2
                    let gpu = runtime.gpuSeconds, start = Date.timeIntervalSinceReferenceDate
                    try runtime.forward([token])
                    samples.append(["split":runtime.splitAttention,"gpu_seconds":runtime.gpuSeconds-gpu,
                                    "wall_seconds":Date.timeIntervalSinceReferenceDate-start])
                }
                XCTAssertEqual(runtime.position,length+32)
                XCTAssertTrue(runtime.logitValues().allSatisfy(\.isFinite))
                try json(["prefix_tokens":length,"samples":samples]).write(to: URL(fileURLWithPath:"results/decode-costs-\(length).json"),options:.atomic)
            }
        }
    }

    func testMaximumContextAndDecode() throws {
        guard ProcessInfo.processInfo.environment["PLAMO_MAX_TEST"] == "1" else { throw XCTSkip("Set PLAMO_MAX_TEST=1 after verify_long.py --lengths 32768") }
        for packed in [false,true] {
            try autoreleasepool { try checkMaximum(packed:packed) }
        }
    }

    private func checkMaximum(packed: Bool) throws {
        let path = URL(fileURLWithPath: "results/reference-long-32768")
        let data = try Data(contentsOf: path.appendingPathExtension("f32"))
        let metadata = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: path.appendingPathExtension("json"))) as? [String:Any])
        let identity = try XCTUnwrap(metadata["identity"] as? [String:String])
        func sha(_ data: Data) -> String { SHA256.hash(data: data).map { String(format:"%02x",$0) }.joined() }
        XCTAssertEqual(sha(data),metadata["sha256"] as? String)
        XCTAssertEqual(identity["teacher_sha256"],"7c9f0066d166c075bdbe83ada3d444e08bd7a6ee3f345f47a1f3c442aa6898c5")
        let tokens = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: "results/reference-long-32768.tokens.json"))) as? [Int])
        let words = tokens.map { Int32($0).littleEndian }
        XCTAssertEqual(words.withUnsafeBytes { sha(Data($0)) },identity["tokens_sha256"])
        XCTAssertEqual(tokens.count,32768)
        XCTAssertEqual(data.count,107520*4)
        var reference = [Float](repeating:0,count:107520)
        _ = reference.withUnsafeMutableBytes { data.copyBytes(to:$0) }
        let runtime = try Runtime(directory:"models/plamo",context:32768,packed:packed)
        _ = try runtime.prefill(Array(tokens.dropLast(32)))
        var samples = [[String:Any]]()
        for (index,token) in tokens.suffix(32).enumerated() {
            runtime.splitAttention = index%4 == 1 || index%4 == 2
            let gpu = runtime.gpuSeconds, start = Date.timeIntervalSinceReferenceDate
            try runtime.forward([token])
            samples.append(["split":runtime.splitAttention,"gpu_seconds":runtime.gpuSeconds-gpu,
                            "wall_seconds":Date.timeIntervalSinceReferenceDate-start])
        }
        XCTAssertEqual(runtime.position,32768)
        let actual = runtime.logitValues()
        XCTAssertTrue(actual.allSatisfy(\.isFinite))
        let error = zip(actual,reference).reduce(0.0) { $0+pow(Double($1.0)-Double($1.1),2) }
        let rmse = sqrt(error/Double(actual.count))
        let dot = zip(actual,reference).reduce(0.0) { $0+Double($1.0)*Double($1.1) }
        let norm = sqrt(actual.reduce(0.0) { $0+Double($1)*Double($1) }*reference.reduce(0.0) { $0+Double($1)*Double($1) })
        let cosine = dot/norm
        func argmax(_ values: [Float]) -> Int { values.indices.max { values[$0] < values[$1] }! }
        XCTAssertEqual(argmax(actual),argmax(reference))
        XCTAssertLessThan(rmse,0.2)
        XCTAssertGreaterThan(cosine,0.999)
        XCTAssertThrowsError(try runtime.forward([500]))
        XCTAssertEqual(runtime.position,32768)
        XCTAssertEqual(actual.map(\.bitPattern),runtime.logitValues().map(\.bitPattern))
        var medians = [String:[String:Double]]()
        for split in [false,true] {
            let warm = Array(samples.filter { $0["split"] as? Bool == split }.dropFirst(2))
            var times = [String:Double]()
            for key in ["gpu_seconds","wall_seconds"] {
                let values = warm.map { $0[key] as! Double }.sorted()
                times[key] = (values[(values.count-1)/2]+values[values.count/2])/2
            }
            medians[split ? "split" : "dense"] = times
        }
        let result: [String:Any] = ["packed":packed,"decode_samples":samples,"decode_medians":medians,"input_tokens":tokens.count,"scalar_tail_tokens":32,"rmse":rmse,"cosine":cosine,
                                  "next_token":argmax(actual),"reference_next_token":argmax(reference),
                                  "gpu_seconds":runtime.gpuSeconds,"metal_allocated_bytes":runtime.device.currentAllocatedSize,
                                  "peak_metal_allocated_bytes":runtime.peakMetalBytes]
        try json(result).write(to: URL(fileURLWithPath:packed ? "results/maximum-decode.json" : "results/maximum-decode-raw.json"),options:.atomic)
    }
}
