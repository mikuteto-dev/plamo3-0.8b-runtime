import XCTest
@testable import Plamo

final class BatchCLITests: XCTestCase {
    func testBatchMatchesSequentialCLI() throws {
        guard let executable = ProcessInfo.processInfo.environment["PLAMO_BATCH_CLI"] else { throw XCTSkip("Set PLAMO_BATCH_CLI to the candidate binary") }
        func run(_ source: String, flags: [String], packed: Bool = true) throws -> (String,[String:Any],Double) {
            let process = Process(), input = Pipe(), output = Pipe(), error = Pipe()
            process.executableURL = URL(fileURLWithPath:executable)
            process.arguments = ["--segment-tokens","64","--to","ja","--max-tokens","64"]+(packed ? ["--packed"] : [])+flags
            process.standardInput = input; process.standardOutput = output; process.standardError = error
            let start = DispatchTime.now().uptimeNanoseconds
            try process.run()
            input.fileHandleForWriting.write(Data(source.utf8));try input.fileHandleForWriting.close()
            let text = output.fileHandleForReading.readDataToEndOfFile(), diagnostic = error.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            XCTAssertEqual(process.terminationStatus,0,String(decoding:diagnostic,as:UTF8.self))
            return (String(decoding:text,as:UTF8.self),try JSONSerialization.jsonObject(with:diagnostic) as! [String:Any],Double(DispatchTime.now().uptimeNanoseconds-start)/1e9)
        }
        let source = "\tThe meeting has been moved to next Monday.\r\n\r\n Hello.\nThank you.\nYes.\nNo.\n  "
        var records = [[String:Any]]()
        for flags in [["--stream"],[],["--stream","--max-tokens","3"]] {
            let scalar = try run(source,flags:flags), batch = try run(source,flags:flags+["--batch4"])
            let sr = (scalar.1["runs"] as! [[String:Any]])[0], br = (batch.1["runs"] as! [[String:Any]])[0]
            let textEqual = scalar.0 == batch.0
            let idsEqual = sr["output_ids"] as! [Int] == br["output_ids"] as! [Int]
            XCTAssertTrue(textEqual);XCTAssertTrue(idsEqual)
            XCTAssertEqual(sr["stop_reason"] as? String,br["stop_reason"] as? String)
            XCTAssertEqual(sr["segments_total"] as? Int,5);XCTAssertEqual(br["segments_total"] as? Int,5)
            XCTAssertFalse(scalar.1["batch4"] as! Bool);XCTAssertTrue(batch.1["batch4"] as! Bool)
            XCTAssertEqual(scalar.1["gpu_weight_bytes"] as! Int,batch.1["gpu_weight_bytes"] as! Int)
            XCTAssertLessThan((batch.1["metal_allocated_bytes"] as! Int)-(scalar.1["metal_allocated_bytes"] as! Int),(batch.1["gpu_weight_bytes"] as! Int)/2)
            XCTAssertGreaterThanOrEqual(batch.1["peak_metal_allocated_bytes"] as! Int,batch.1["metal_allocated_bytes"] as! Int)
            if flags.contains("3") {
                XCTAssertEqual(br["stop_reason"] as? String,"length")
                XCTAssertEqual((br["segments"] as! [[String:Any]]).count,1)
                XCTAssertEqual(br["segments_completed"] as? Int,0)
                XCTAssertGreaterThan(br["discarded_output_tokens"] as! Int,0)
                XCTAssertGreaterThan(br["prefilled_input_tokens"] as! Int,br["input_tokens"] as! Int)
            } else {
                XCTAssertEqual(br["stop_reason"] as? String,"stop")
                XCTAssertEqual(br["segments_completed"] as? Int,5)
                XCTAssertEqual(br["discarded_output_tokens"] as? Int,0)
                XCTAssertEqual(br["prefilled_input_tokens"] as? Int,br["input_tokens"] as? Int)
            }
            records.append(["flags":flags,"stdout_equal":textEqual,"ids_equal":idsEqual,"stop_reason":br["stop_reason"]!,
                            "scalar_process_seconds":scalar.2,"batch_process_seconds":batch.2])
        }
        let expected = try run(source,flags:[]).0
        let repeated = try run(source,flags:["--batch4","--repeat","2"])
        XCTAssertEqual(repeated.0,expected)
        let runs = repeated.1["runs"] as! [[String:Any]]
        XCTAssertEqual(runs.count,2)
        XCTAssertEqual(runs[0]["output_ids"] as! [Int],runs[1]["output_ids"] as! [Int])
        var singleCases = [[String:Any]]()
        for packed in [false,true] {
            for flags in [["--stream","--max-tokens","2048"],["--repeat","2","--max-tokens","2048"],
                          ["--stream","--max-tokens","3","--context","8191"]] {
                let scalar = try run("\tThe meeting has been moved to next Monday.\r\n ",flags:flags,packed:packed)
                let fast = try run("\tThe meeting has been moved to next Monday.\r\n ",flags:flags+["--batch4"],packed:packed)
                let sr = scalar.1["runs"] as! [[String:Any]], br = fast.1["runs"] as! [[String:Any]]
                let metalEqual = scalar.1["metal_allocated_bytes"] as! Int == fast.1["metal_allocated_bytes"] as! Int
                XCTAssertEqual(scalar.0,fast.0);XCTAssertTrue(metalEqual)
                XCTAssertEqual(sr.count,br.count);XCTAssertTrue(fast.1["batch4"] as! Bool)
                for (s,b) in zip(sr,br) {
                    XCTAssertEqual(s["output_ids"] as! [Int],b["output_ids"] as! [Int])
                    XCTAssertEqual(s["stop_reason"] as! String,b["stop_reason"] as! String)
                    XCTAssertEqual(b["segments_total"] as! Int,1)
                    XCTAssertEqual(b["prefilled_input_tokens"] as! Int,b["input_tokens"] as! Int)
                    XCTAssertEqual(b["discarded_output_tokens"] as! Int,0)
                }
                singleCases.append(["packed":packed,"flags":flags,"stdout_equal":scalar.0==fast.0,
                                    "ids_equal":zip(sr,br).allSatisfy { $0.0["output_ids"] as! [Int] == $0.1["output_ids"] as! [Int] },
                                    "metal_equal":metalEqual,"scalar_metal_bytes":scalar.1["metal_allocated_bytes"]!,
                                    "batch_flag_metal_bytes":fast.1["metal_allocated_bytes"]!,
                                    "scalar_process_seconds":scalar.2,"batch_flag_process_seconds":fast.2])
            }
        }
        try json(["cases":records,"single_segment_cases":singleCases,"repeats_match":runs[0]["output_ids"] as! [Int] == runs[1]["output_ids"] as! [Int],
                  "timing_scope":"Cold process checks for a short five-document fixture, not a statistically controlled benchmark", "default_batching_enabled":false] as [String:Any])
            .write(to:URL(fileURLWithPath:"results/batch-cli-correctness.json"),options:.atomic)
    }
}
