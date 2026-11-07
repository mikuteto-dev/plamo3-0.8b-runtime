import XCTest
@testable import Plamo

final class SegmentTests: XCTestCase {
    private func tokenizer() throws -> Tokenizer { try Tokenizer(path:"models/plamo/tokenizer.jsonl") }

    func testWhitespaceAndDocumentBoundaries() throws {
        let tokenizer = try tokenizer()
        let text = "\tFirst document.\r\n\r\n  第二文書。 🧪e\u{301}\u{2029}Third document.\n  "
        let parts = try Segment.split(text,target:"ja",limit:64,tokenizer:tokenizer)
        XCTAssertEqual(parts.map(\.text).joined(),text)
        XCTAssertEqual(parts.filter { !$0.tokens.isEmpty }.count,3)
        for part in parts where !part.tokens.isEmpty {
            XCTAssertEqual(part.tokens,tokenizer.encode(try Tokenizer.prompt(part.text,target:"ja"),bos:true))
            XCTAssertLessThanOrEqual(part.tokens.count,64)
            XCTAssertFalse(part.text.first!.isWhitespace)
            XCTAssertFalse(part.text.last!.isWhitespace)
        }
    }

    func testManyBlankLinesStayLiteral() throws {
        let text = String(repeating:"\r\n \t",count:3000)+"Hello."
        let parts = try Segment.split(text,target:"ja",limit:64,tokenizer:tokenizer())
        XCTAssertEqual(parts.map(\.text).joined(),text)
        XCTAssertEqual(parts.count,2)
        XCTAssertTrue(parts[0].tokens.isEmpty)
        XCTAssertFalse(parts[1].tokens.isEmpty)
    }

    func testTokenBoundAndUnbrokenText() throws {
        let tokenizer = try tokenizer()
        for text in [String(repeating:"A short sentence. 次の文です。 ",count:120),
                     String(repeating:"𠮷🧪e\u{301}",count:180),String(repeating:"abcdef",count:250)] {
            let parts = try Segment.split(text,target:"en",limit:64,tokenizer:tokenizer)
            XCTAssertEqual(parts.map(\.text).joined(),text)
            XCTAssertGreaterThan(parts.filter { !$0.tokens.isEmpty }.count,1)
            XCTAssertTrue(parts.allSatisfy { $0.tokens.count <= 64 })
        }
        let short = try Segment.split("First sentence. Second sentence.",target:"ja",limit:64,tokenizer:tokenizer)
        XCTAssertEqual(short.count,1)
    }

    func testRejectsInvalidAndUnsplittableInput() throws {
        let tokenizer = try tokenizer()
        for text in ["", " \t\r\n", "safe\n<|plamo:tag|>", "a\u{ee00}b"] {
            XCTAssertThrowsError(try Segment.split(text,target:"en",limit:64,tokenizer:tokenizer))
        }
        XCTAssertThrowsError(try Segment.split("text",target:"invalid",limit:64,tokenizer:tokenizer))
        XCTAssertThrowsError(try Segment.split("text",target:"en",limit:0,tokenizer:tokenizer))
        let grapheme = "a"+String(repeating:"\u{301}",count:800)
        XCTAssertEqual(grapheme.count,1)
        XCTAssertThrowsError(try Segment.split(grapheme,target:"en",limit:64,tokenizer:tokenizer))
    }

    func testOptionBoundaries() throws {
        for args in [["--segment-tokens","0"],["--segment-tokens","63"],["--segment-tokens","8193"],
                     ["--segment-tokens","64","--tokenize","text"],["--segment-tokens","64","--tokens","1"],
                     ["--segment-tokens","64","--decode","1"],["--segment-tokens","64","--dump-logits","x"]] {
            XCTAssertThrowsError(try Options(args))
        }
        XCTAssertEqual(try Options(["--segment-tokens","512","--stream"]).segmentTokens,512)
    }

    func testPipelineMatchesIndependentTranslations() throws {
        guard ProcessInfo.processInfo.environment["PLAMO_SEGMENTS"] == "1" else { throw XCTSkip("Set PLAMO_SEGMENTS=1 after swift build -c release") }
        func run(_ text: String, _ args: [String] = []) throws -> (String,[String:Any]) {
            let process = Process(), input = Pipe(), output = Pipe(), error = Pipe()
            process.executableURL = URL(fileURLWithPath:".build/release/plamo")
            process.arguments = ["--packed","--to","ja","--max-tokens","64"]+args
            process.standardInput = input; process.standardOutput = output; process.standardError = error
            try process.run()
            input.fileHandleForWriting.write(Data(text.utf8)); try input.fileHandleForWriting.close()
            let translated = output.fileHandleForReading.readDataToEndOfFile()
            let diagnostic = error.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            XCTAssertEqual(process.terminationStatus,0,String(decoding:diagnostic,as:UTF8.self))
            return (String(decoding:translated,as:UTF8.self),try JSONSerialization.jsonObject(with:diagnostic) as! [String:Any])
        }
        let texts = ["Good morning.","The meeting starts at ten.","Thank you."]
        let translations = try texts.map { try run($0,["--stream"]).0 }
        let source = "\t"+texts[0]+" \r\n\r\n "+texts[1]+"\n"+texts[2]+"\n"
        let expected = "\t"+translations[0]+" \r\n\r\n "+translations[1]+"\n"+translations[2]+"\n"
        for args in [["--stream"],["--repeat","2"]] {
            let (output,metrics) = try run(source,["--segment-tokens","64"]+args)
            XCTAssertEqual(output,expected+(args.contains("--stream") ? "" : "\n"))
            let runs = metrics["runs"] as! [[String:Any]]
            XCTAssertEqual(runs.count,args.contains("--stream") ? 1 : 2)
            for result in runs {
                XCTAssertEqual(result["segments_total"] as? Int,3)
                XCTAssertEqual(result["segments_completed"] as? Int,3)
                XCTAssertEqual(result["stop_reason"] as? String,"stop")
                let segments = result["segments"] as! [[String:Any]]
                XCTAssertEqual(segments.count,3)
                XCTAssertEqual(result["output_tokens"] as? Int,segments.reduce(0) { $0+($1["output_tokens"] as! Int) })
            }
        }
        let unbroken = String(repeating:"会議は明日です。資料はメールで送ります。参加者は十人です。",count:8)
        let parts = try Segment.split(unbroken,target:"en",limit:64,tokenizer:tokenizer())
        XCTAssertGreaterThan(parts.count,1)
        XCTAssertTrue(parts.allSatisfy { !$0.tokens.isEmpty })
        let expectedChunks = try parts.map { try run($0.text,["--stream","--to","en","--max-tokens","256"]).0 }
        let (chunked,_) = try run(unbroken,["--stream","--to","en","--max-tokens","256","--segment-tokens","64"])
        XCTAssertEqual(chunked,expectedChunks.joined(separator:"\n"))
        let (_,metrics) = try run(source,["--segment-tokens","64","--max-tokens","1"])
        let result = (metrics["runs"] as! [[String:Any]])[0]
        XCTAssertEqual(result["stop_reason"] as? String,"length")
        XCTAssertEqual(result["segments_completed"] as? Int,0)
        XCTAssertEqual((result["segments"] as! [[String:Any]]).count,1)
        XCTAssertEqual(result["segments_total"] as? Int,3)
    }
}
