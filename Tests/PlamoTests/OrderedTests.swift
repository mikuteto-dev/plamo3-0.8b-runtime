import XCTest
@testable import Plamo

final class OrderedTests: XCTestCase {
    private enum Cancelled: Error { case requested }
    private func tokenizer() throws -> Tokenizer { try Tokenizer(path:"models/plamo/tokenizer.jsonl") }
    private func bytes(_ text: String) -> [Int] { text.utf8.map { $0 < 128 ? Int($0)+384 : Int($0) } }

    private func simulate(_ texts: [String], limits: [Int], prefixes: [String], tail: String = "") throws -> (String,[BatchOutput],[[Int?]],[String]) {
        let source = texts.map(bytes), tokenizer = try tokenizer()
        var output = "", cursor = [Int](repeating:0,count:texts.count), calls = [[Int?]](), seen = [String]()
        let sink = try OrderedOutput(tokenizer:tokenizer,prefixes:prefixes,tail:tail,emit:{ output += $0 })
        func next() -> [Int] { source.indices.map { cursor[$0] < source[$0].count ? source[$0][cursor[$0]] : ($0%2 == 0 ? 1 : 16) } }
        let results = try decodeBatch(next(),limits:limits,step:{ tokens in
            calls.append(tokens); seen.append(output)
            for lane in tokens.indices where tokens[lane] != nil {
                XCTAssertEqual(tokens[lane],source[lane][cursor[lane]])
                cursor[lane] += 1
            }
            return next()
        },emit:{ try sink.append($0,$1) },finish:{ try sink.finish($0,$1) },stopOnTruncation:true)
        return (output,results,calls,seen)
    }

    func testFutureEOSAndUTF8StayInDocumentOrder() throws {
        let (text,outputs,_,seen) = try simulate(["AAA","B","あ",""],limits:[20,20,20,20],prefixes:["  ","\r\n\r\n","\t","\n"],tail:"\r\n ")
        XCTAssertEqual(text,"  AAA\r\n\r\nB\tあ\n\r\n ")
        XCTAssertEqual(outputs.map(\.reason),["stop","stop","stop","stop"])
        XCTAssertEqual(seen,["  A","  AA","  AAA"])
        XCTAssertEqual(try tokenizer().decode(outputs[2].ids),"あ")
    }

    func testLaterTruncationKeepsEarlierDocumentAndDropsFuture() throws {
        let (text,outputs,calls,_) = try simulate(["AAA","BBBB","C","DDDD"],limits:[20,2,20,20],prefixes:["<0>","<1>","<2>","<3>"],tail:"<tail>")
        XCTAssertEqual(text,"<0>AAA<1>BB")
        XCTAssertEqual(outputs.map(\.reason),["stop","length"])
        XCTAssertEqual(outputs.map(\.ids),[bytes("AAA"),bytes("BB")])
        XCTAssertEqual(calls[0],[449,450,451,452])
        XCTAssertTrue(calls.dropFirst().allSatisfy { $0[1] == nil && $0[2] == nil && $0[3] == nil })
        let early = try simulate(["AAAA","BBBB","CCCC","D"],limits:[2,20,1,20],prefixes:["","<1>","<2>","<3>"],tail:"<tail>")
        XCTAssertEqual(early.0,"AA")
        XCTAssertEqual(early.1.map(\.reason),["length"])
        let zero = try simulate(["A","B"],limits:[0,20],prefixes:["prefix","later"],tail:"tail")
        XCTAssertEqual(zero.0,"prefix"); XCTAssertTrue(zero.2.isEmpty)
    }

    func testPlanPreservesWhitespaceAndAdjacentChunkBoundaries() throws {
        let parts = [Segment(text:"\t",tokens:[]),Segment(text:"source A",tokens:[449]),
                     Segment(text:"\r\n",tokens:[]),Segment(text:" \n",tokens:[]),Segment(text:"source B",tokens:[450]),
                     Segment(text:"source C",tokens:[451]),Segment(text:"\n\t",tokens:[])]
        let plan = try DocumentPlan(parts)
        XCTAssertEqual(plan.inputs,[[449],[450],[451]])
        XCTAssertEqual(plan.prefixes,["\t","\r\n \n","\n"])
        XCTAssertEqual(plan.tail,"\n\t")
        XCTAssertThrowsError(try DocumentPlan([]))
        XCTAssertThrowsError(try DocumentPlan([Segment(text:" \n",tokens:[])]))
    }

    func testFinishNotificationsPreserveIndependentMode() throws {
        var calls = [[Int?]](), finished = [Int](), pending = [[1,16,100,201],[1,16,100,202]]
        let outputs = try decodeBatch([1,16,100,200],limits:[9,9,1,3],step:{ calls.append($0);return pending.removeFirst() },finish:{ lane,_ in finished.append(lane) })
        XCTAssertEqual(outputs.map(\.ids),[[],[],[100],[200,201,202]])
        XCTAssertEqual(outputs.map(\.reason),["stop","stop","length","length"])
        XCTAssertEqual(calls,[[nil,nil,nil,200],[nil,nil,nil,201]])
        XCTAssertEqual(finished,[0,1,2,3])
    }

    func testIncompleteFutureUTF8AndFinishCancellation() throws {
        let t = try tokenizer()
        var text = ""
        let sink = try OrderedOutput(tokenizer:t,prefixes:["","\n"],tail:"tail",emit:{ text += $0 })
        try sink.append(1,227)
        try sink.finish(1,BatchOutput(ids:[227],reason:"stop"))
        XCTAssertEqual(text,"")
        try sink.append(0,449)
        try sink.finish(0,BatchOutput(ids:[449],reason:"stop"))
        XCTAssertEqual(text,"A\n\u{fffd}tail")
        var steps = 0
        let cancelled = try OrderedOutput(tokenizer:t,prefixes:["","next"],tail:"",emit:{ _ in throw Cancelled.requested })
        XCTAssertThrowsError(try decodeBatch([1,450],limits:[20,20],step:{ _ in steps += 1;return [1,16] },
                                             emit:{ try cancelled.append($0,$1) },finish:{ try cancelled.finish($0,$1) },stopOnTruncation:true)) {
            XCTAssertTrue($0 is Cancelled)
        }
        XCTAssertEqual(steps,0)
    }

    func testBatchOptionBoundaries() throws {
        for args in [["--batch4"],["--batch4","--segment-tokens","64","--qsa"],
                     ["--batch4","--segment-tokens","64","--mtp"],
                     ["--batch4","--segment-tokens","64","--repetition-penalty","1.1"],
                     ["--batch4","--segment-tokens","64","--frequency-penalty","0.1"],
                     ["--batch4","--segment-tokens","64","--no-repeat-ngram","1"],
                     ["--batch4","--segment-tokens","64","--loop-limit","4"],
                     ["--batch4","--segment-tokens","64","--context","8192"]] {
            XCTAssertThrowsError(try Options(args))
        }
        XCTAssertTrue(try Options(["--batch4","--segment-tokens","512","--stream"]).batch4)
        XCTAssertFalse(try Options([]).batch4)
    }

    func testCancellationAndInvalidOutputEvents() throws {
        let t = try tokenizer()
        var text = "", steps = 0
        let sink = try OrderedOutput(tokenizer:t,prefixes:["","next"],tail:"tail",emit:{ text += $0;throw Cancelled.requested })
        XCTAssertThrowsError(try decodeBatch([449,450],limits:[20,20],step:{ _ in steps += 1;return [1,16] },emit:{ try sink.append($0,$1) },finish:{ try sink.finish($0,$1) },stopOnTruncation:true)) {
            XCTAssertTrue($0 is Cancelled)
        }
        XCTAssertEqual(text,"A");XCTAssertEqual(steps,0)
        let valid = try OrderedOutput(tokenizer:t,prefixes:[""],tail:"",emit:{ _ in })
        XCTAssertThrowsError(try valid.append(1,449))
        XCTAssertThrowsError(try valid.append(0,107520))
        try valid.finish(0,BatchOutput(ids:[],reason:"stop"))
        XCTAssertThrowsError(try valid.append(0,449))
        XCTAssertThrowsError(try valid.finish(0,BatchOutput(ids:[],reason:"stop")))
    }

    private func scalar(_ source: Runtime, prompt: [Int], limit: Int) throws -> BatchOutput {
        source.reset()
        var next = try source.prefill(prompt), ids = [Int]()
        for i in 0..<limit {
            if next == 1 || next == 16 { return BatchOutput(ids:ids,reason:"stop") }
            ids.append(next)
            if i+1 < limit { next = try source.forward([next]) }
        }
        return BatchOutput(ids:ids,reason:"length")
    }

    func testActualMultipleGroupsAndTruncationMatchSequentialOutput() throws {
        guard ProcessInfo.processInfo.environment["PLAMO_ORDERED"] == "1" else { throw XCTSkip("Set PLAMO_ORDERED=1") }
        let t = try tokenizer(), batch = try Batch(Runtime(directory:"models/plamo",context:128,packed:true))
        let source = "\tThe meeting has been moved to next Monday.\r\n\r\n Hello.\nThank you.\nYes.\nNo.\n  "
        let parts = try Segment.split(source,target:"ja",limit:64,tokenizer:t), plan = try DocumentPlan(parts)
        XCTAssertEqual(plan.inputs.count,5)
        let expected = try plan.inputs.map { try scalar(batch.scalar,prompt:$0,limit:64) }
        XCTAssertTrue(expected.allSatisfy { $0.reason == "stop" })
        var expectedText = ""
        for lane in expected.indices { expectedText += plan.prefixes[lane]+(try t.decode(expected[lane].ids)) }
        expectedText += plan.tail
        var text = ""
        let actual = try batch.translateOrdered(parts,tokenizer:t,limit:64,emit:{ text += $0 })
        XCTAssertEqual(text,expectedText);XCTAssertEqual(actual.sequences.map(\.ids),expected.map(\.ids))
        XCTAssertEqual(actual.sequences.map(\.reason),expected.map(\.reason));XCTAssertEqual(actual.discardedTokens,0)
        // The later short document hits its cap before the longer first document finishes.
        let first = try scalar(batch.scalar,prompt:plan.inputs[0],limit:64)
        let second = try scalar(batch.scalar,prompt:plan.inputs[1],limit:1)
        var partial = "", generated = 0
        let sink = try OrderedOutput(tokenizer:t,prefixes:Array(plan.prefixes.prefix(4)),tail:"SHOULD NOT APPEAR",emit:{ partial += $0 })
        let stopped = try batch.generate(Array(plan.inputs.prefix(4)),limits:[64,1,64,64],emit:{ lane,id in generated += 1;try sink.append(lane,id) },finish:{ try sink.finish($0,$1) },stopOnTruncation:true)
        XCTAssertEqual(partial,plan.prefixes[0]+(try t.decode(first.ids))+plan.prefixes[1]+(try t.decode(second.ids)))
        XCTAssertEqual(stopped.map(\.ids),[first.ids,second.ids]);XCTAssertEqual(stopped.map(\.reason),["stop","length"])
        XCTAssertGreaterThanOrEqual(generated,stopped.reduce(0) { $0+$1.ids.count })
        text = ""
        let capped = try batch.translateOrdered(parts,tokenizer:t,limit:1,emit:{ text += $0 })
        XCTAssertEqual(capped.sequences.count,1);XCTAssertEqual(capped.sequences[0].reason,"length")
        XCTAssertEqual(text,plan.prefixes[0]+(try t.decode(capped.sequences[0].ids)))
        XCTAssertThrowsError(try batch.translateOrdered(parts,tokenizer:t,limit:64,emit:{ _ in throw Cancelled.requested })) {
            XCTAssertTrue($0 is Cancelled)
        }
        XCTAssertFalse(batch.ready);XCTAssertEqual(batch.predictions(),[])
        text = ""
        XCTAssertEqual(try batch.translateOrdered(parts,tokenizer:t,limit:64,emit:{ text += $0 }).sequences.map(\.ids),expected.map(\.ids))
        XCTAssertEqual(text,expectedText)
        var emitted = false
        XCTAssertThrowsError(try batch.translateOrdered([Segment(text:"",tokens:[107520])],tokenizer:t,limit:64,emit:{ _ in emitted = true }))
        XCTAssertFalse(emitted)
        let validation = ProcessInfo.processInfo.environment["MTL_SHADER_VALIDATION"] == "1"
        try json(["five_document_ids_match":actual.sequences.map(\.ids) == expected.map(\.ids),"order_whitespace_matches":text==expectedText,
                  "truncation_output_ids":stopped.map(\.ids),"truncation_reasons":stopped.map(\.reason),"shader_validation":validation,"default_batching_enabled":false] as [String:Any])
            .write(to:URL(fileURLWithPath:"results/ordered-\(validation ? "validation" : "correctness").json"),options:.atomic)
    }
}
