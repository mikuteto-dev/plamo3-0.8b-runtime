import XCTest
import CryptoKit
@testable import Plamo

final class VerificationTests: XCTestCase {
    func testBatchedVerificationAndRollback() throws {
        guard FileManager.default.fileExists(atPath:"models/plamo/auxiliary.safetensors") else { throw XCTSkip("Train adapters first") }
        let lengths = ProcessInfo.processInfo.environment["PLAMO_LONG_TEST"] == "1" ? [31,2047,2063] : [31]
        for (length,budget) in lengths.flatMap({ n in [(n,Optional<Int>.none),(n,Optional(16))] }) {
            for packed in [false,true] {
                try autoreleasepool { try checkPair(length,budget:budget,packed:packed) }
            }
        }
    }

    private func checkPair(_ length: Int, budget: Int?, packed: Bool) throws {
        let scalar = try Runtime(directory: "models/plamo", context: length+32, packed: packed, qsaBudget: budget, mtp: true)
        let batched = try Runtime(directory: "models/plamo", context: length+32, packed: packed, qsaBudget: budget, mtp: true)
        let prefix = [1]+(0..<(length-1)).map { 500+$0%37 }
        XCTAssertEqual(try scalar.prefill(prefix), try batched.prefill(prefix))
        let before = batched.logitValues().map(\.bitPattern)
        for (all,sequential) in [(true,false),(false,true),(true,true)] {
            XCTAssertThrowsError(try batched.forward([600,700,800],all:all,sequential:sequential))
            XCTAssertEqual(batched.position,length)
            XCTAssertEqual(batched.logitValues().map(\.bitPattern),before)
        }
        let inputs = [600,700]
        var expected = [[UInt32]](), next = [Int]()
        for token in inputs {
            next.append(try scalar.forward([token]))
            expected.append(scalar.logitValues().map(\.bitPattern))
        }
        try batched.forward(inputs, all: true, sequential: true)
        XCTAssertEqual(batched.predictions(), next)
        for i in inputs.indices {
            XCTAssertEqual(batched.logitValues(slot: i).map(\.bitPattern), expected[i], "position=\(length+i)")
        }
        try batched.rewind(length+2,hiddenIndex:1)
        XCTAssertEqual(batched.position,length+2)
        XCTAssertEqual(batched.predictions(),[next[1]])
        XCTAssertEqual(batched.logitValues().map(\.bitPattern),expected[1])
        XCTAssertEqual(try batched.forward([901]),try scalar.forward([901]))
        XCTAssertEqual(batched.logitValues().map(\.bitPattern),scalar.logitValues().map(\.bitPattern))
        batched.reset()
        _ = try batched.prefill(prefix)
        try batched.forward(inputs,all:true,sequential:true)
        try batched.rewind(length+1, hiddenIndex: 0)
        let clean = try Runtime(directory: "models/plamo", context: length+32, packed: packed, qsaBudget: budget, mtp: true)
        _ = try clean.prefill(prefix)
        try clean.forward([inputs[0]])
        XCTAssertEqual(try batched.forward([901]), try clean.forward([901]))
        XCTAssertEqual(batched.logitValues().map(\.bitPattern), clean.logitValues().map(\.bitPattern))
        XCTAssertThrowsError(try batched.rewind(0, hiddenIndex: 9))
    }

    func testMaximumPair() throws {
        guard ProcessInfo.processInfo.environment["PLAMO_MAX_TEST"] == "1" else { throw XCTSkip("Set PLAMO_MAX_TEST=1 after maximum reference generation") }
        let prefix = "results/reference-long-32768"
        let data = try Data(contentsOf:URL(fileURLWithPath:prefix+".f32"))
        let metadata = try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:prefix+".json"))) as? [String:Any])
        let identity = try XCTUnwrap(metadata["identity"] as? [String:String])
        let tokens = try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:prefix+".tokens.json"))) as? [Int])
        func sha(_ data: Data) -> String { SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined() }
        guard data.count == 107520*4, tokens.count == 32768, tokens.allSatisfy({ (0..<107520).contains($0) }) else {
            throw Failure.message("Invalid maximum-context fixture")
        }
        let words = tokens.map { Int32($0).littleEndian }
        guard sha(data) == metadata["sha256"] as? String,
              words.withUnsafeBytes({ sha(Data($0)) }) == identity["tokens_sha256"],
              identity["teacher_sha256"] == "7c9f0066d166c075bdbe83ada3d444e08bd7a6ee3f345f47a1f3c442aa6898c5" else {
            throw Failure.message("Stale maximum-context fixture")
        }
        var reference = [Float](repeating:0,count:107520)
        _ = reference.withUnsafeMutableBytes { data.copyBytes(to:$0) }
        let model = try Runtime(directory:"models/plamo",context:32768,packed:true,mtp:true)
        _ = try model.prefill(Array(tokens.dropLast(2)))
        let draft = try model.draft(tokens[tokens.count-2])
        XCTAssertEqual(draft.count,1)
        XCTAssertTrue(draft.allSatisfy { (0..<107520).contains($0) })
        XCTAssertEqual(model.position,32766)
        try model.forward(Array(tokens.suffix(2)),all:true,sequential:true)
        let pair = model.logitValues(slot:1), predictions = model.predictions()
        XCTAssertEqual(model.position,32768)
        XCTAssertTrue(pair.allSatisfy(\.isFinite))
        let next = try XCTUnwrap(reference.indices.max { reference[$0] < reference[$1] })
        var error = 0.0, dot = 0.0, actualNorm = 0.0, referenceNorm = 0.0
        for (actual,expected) in zip(pair,reference) {
            let a = Double(actual), b = Double(expected)
            error += (a-b)*(a-b); dot += a*b; actualNorm += a*a; referenceNorm += b*b
        }
        let rmse = sqrt(error/Double(pair.count)), cosine = dot/sqrt(actualNorm*referenceNorm)
        XCTAssertEqual(predictions.last,next)
        XCTAssertLessThan(rmse,0.2)
        XCTAssertGreaterThan(cosine,0.999)
        try model.rewind(32767,hiddenIndex:0)
        try model.forward([tokens.last!])
        XCTAssertTrue(pair.map(\.bitPattern).elementsEqual(model.logitValues().map(\.bitPattern)))
        XCTAssertEqual(model.predictions(),[next])
        XCTAssertThrowsError(try model.forward([500]))
        XCTAssertEqual(model.position,32768)
        XCTAssertTrue(pair.map(\.bitPattern).elementsEqual(model.logitValues().map(\.bitPattern)))
        try json(["input_tokens":tokens.count,"verify_rows":Runtime.verifyRows,"rmse":rmse,"cosine":cosine,
                  "next_token":next,"draft_checked":true,"scalar_replay_bitwise_equal":true,
                  "metal_allocated_bytes":model.device.currentAllocatedSize,"peak_metal_allocated_bytes":model.peakMetalBytes])
            .write(to:URL(fileURLWithPath:"results/maximum-mtp-pair.json"),options:.atomic)
    }

    func testTrainedDraftShapes() throws {
        guard FileManager.default.fileExists(atPath: "models/plamo/auxiliary.safetensors") else { throw XCTSkip("Train adapters first") }
        let runtime = try Runtime(directory: "models/plamo", context: 64, packed: true, mtp: true)
        XCTAssertGreaterThanOrEqual(runtime.result.buffer.length,Runtime.verifyRows*4)
        XCTAssertThrowsError(try runtime.draft(500))
        _ = try runtime.prefill([1,500,600])
        for token in [-1,107520,1,16] { XCTAssertThrowsError(try runtime.draft(token)) }
        let before = runtime.position
        let drafts = try runtime.draft(500)
        XCTAssertEqual(drafts.count, 1)
        XCTAssertTrue(drafts.allSatisfy { (0..<107520).contains($0) })
        XCTAssertEqual(runtime.position, before)
    }
}
