import XCTest
import Metal
@testable import Plamo

final class PlamoTests: XCTestCase {
    func testArgumentBoundaries() throws {
        for args in [["--max-tokens", "0"], ["--context", "32769"], ["--repeat", "0"],
                     ["--to"], ["--unknown", "x"], ["first", "second"], ["--tokens", "1", "text"], ["--qsa-budget", "0"], ["--qsa-budget", "5"], ["--qsa-budget", "2052"], ["--stream", "--repeat", "2"], ["--stream", "--decode", "1"], ["--stream", "--tokenize", "x"],
                     ["--repetition-penalty", "0"], ["--repetition-penalty", "-1"], ["--frequency-penalty", "-1"],
                     ["--no-repeat-ngram", "65"], ["--loop-limit", "-1"], ["--mtp", "--loop-limit", "4"]] {
            XCTAssertThrowsError(try Options(args), "\(args)")
        }
        XCTAssertThrowsError(try Options(["--max-tokens","0","--tokens","1"]))
        XCTAssertEqual(try Options(["--max-tokens","0","--tokens","1","--dump-logits","test.f32"]).maxTokens,0)
        for text in ["", "1,", "-1", "107520", "1,no"] { XCTAssertThrowsError(try parseIDs(text)) }
        XCTAssertEqual(try parseIDs("1,107519"), [1,107519])
    }

    func testPromptBoundaries() throws {
        XCTAssertThrowsError(try Tokenizer.prompt("", target: "en"))
        XCTAssertThrowsError(try Tokenizer.prompt("text", target: "invalid"))
        XCTAssertThrowsError(try Tokenizer.prompt("<|plamo:tag|>", target: "en"))
        XCTAssertThrowsError(try Tokenizer.prompt("a\u{ee00}b", target: "en"))
        XCTAssertThrowsError(try Tokenizer.prompt("\u{ee00}\u{301}x", target: "en"))
        XCTAssertTrue(try Tokenizer.prompt("hello", target: "ja").hasSuffix("ja<|plamo:msg|>"))
    }

    func testTokenizerRoundTrips() throws {
        let path = "models/plamo/tokenizer.jsonl"
        guard FileManager.default.fileExists(atPath: path) else { throw XCTSkip("Model vocabulary not downloaded") }
        let t = try Tokenizer(path: path)
        for text in ["", "Hello, world!", "会議は来週の月曜日です。", " a  b    c", "aaaaaa\n\n\n\n", "🧪𠮷\u{10ffff}", "<|plamo:tag|>input<|plamo:msg|>hello", "é e\u{301}\r\n\t", "<|plamo:tag|>\u{301}x", "a    \u{301}b", "\u{301}\u{301}\u{301}\u{301}"] {
            XCTAssertEqual(try t.decode(t.encode(text)), text)
        }
        XCTAssertEqual(t.encode("<|plamo:bos|>"), [1])
        XCTAssertThrowsError(try t.decode([-1]))
        XCTAssertThrowsError(try t.decode([107520]))
    }

    func testStreamingDecodeMatchesFullDecode() throws {
        let path = "models/plamo/tokenizer.jsonl"
        guard FileManager.default.fileExists(atPath: path) else { throw XCTSkip("Model vocabulary not downloaded") }
        let tokenizer = try Tokenizer(path: path)
        let ids = tokenizer.encode("会議は来週の月曜日です。🧪𠮷 stream check")
        let stream = Tokenizer.Stream(tokenizer)
        var text = ""
        for id in ids { text += try stream.push(id) }
        text += stream.finish()
        XCTAssertEqual(text, try tokenizer.decode(ids))
        // A multi-byte character split across byte tokens is held back until it is complete.
        let split = Tokenizer.Stream(tokenizer)
        XCTAssertEqual(try split.push(227), "")
        XCTAssertEqual(try split.push(129), "")
        XCTAssertEqual(try split.push(130), "あ")
        XCTAssertEqual(split.finish(), "")
        XCTAssertEqual(try tokenizer.decode([227, 129, 130]), "あ")
        XCTAssertThrowsError(try Tokenizer.Stream(tokenizer).push(107520))
    }

    func testTokenizerScoreChoice() throws {
        let source = "models/plamo/tokenizer.jsonl"
        guard FileManager.default.fileExists(atPath:source) else { throw XCTSkip("Download vocabulary first") }
        let tokenizer = try Tokenizer(path:source)
        let a = try XCTUnwrap(tokenizer.pieces.firstIndex(of:"a"))
        let b = try XCTUnwrap(tokenizer.pieces.firstIndex(of:"b"))
        let ab = try XCTUnwrap(tokenizer.pieces.firstIndex(of:"ab"))
        var lines = try String(contentsOfFile:source,encoding:.utf8).split(separator:"\n").map(String.init)
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:path) }
        lines[a] = "[\"a\",300000,\"NORMAL\"]"
        lines[b] = "[\"b\",300000,\"NORMAL\"]"
        for (score,expected) in [(500000,[a,b]),(600000,[ab]),(700000,[ab])] {
            lines[ab] = "[\"ab\",\(score),\"NORMAL\"]"
            try lines.joined(separator:"\n").write(to:path,atomically:true,encoding:.utf8)
            let candidate = try Tokenizer(path:path.path)
            XCTAssertEqual(candidate.encode("ab"),expected)
            XCTAssertEqual(candidate.encode("aba"),expected+[a])
            XCTAssertEqual(candidate.encode("babab"),[b]+expected+expected)
        }
    }

    func testTokenSpanExceedsUInt16() throws {
        let source = "models/plamo/tokenizer.jsonl"
        guard FileManager.default.fileExists(atPath:source) else { throw XCTSkip("Download vocabulary first") }
        var lines = try String(contentsOfFile:source,encoding:.utf8).split(separator:"\n").map(String.init)
        let word = "\u{10fffd}"+String(repeating:"ab",count:32768)
        XCTAssertGreaterThan(word.unicodeScalars.count,Int(UInt16.max))
        let record = try JSONSerialization.data(withJSONObject:[word,100000,"NORMAL"])
        lines[107519] = String(decoding:record,as:UTF8.self)
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:path) }
        try lines.joined(separator:"\n").write(to:path,atomically:true,encoding:.utf8)
        let tokenizer = try Tokenizer(path:path.path)
        XCTAssertEqual(tokenizer.encode(word),[107519])
        XCTAssertEqual(try tokenizer.decode([107519]),word)
        let repeated = String(repeating:word,count:3)
        XCTAssertEqual(tokenizer.encode(repeated),[107519,107519,107519])
        XCTAssertEqual(try tokenizer.decode([107519,107519,107519]),repeated)
    }

    func testVocabularyRecordsMatchOriginal() throws {
        let path = "models/plamo/tokenizer.jsonl"
        guard FileManager.default.fileExists(atPath:path) else { throw XCTSkip("Download vocabulary first") }
        let lines = try String(contentsOfFile:path,encoding:.utf8).split(separator:"\n")
        XCTAssertEqual(lines.count,107520)
        let decoder = JSONDecoder()
        for start in stride(from:0,to:lines.count,by:512) {
            try autoreleasepool {
                for line in lines[start..<min(start+512,lines.count)] {
                    let data = Data(line.utf8)
                    let old = try XCTUnwrap(JSONSerialization.jsonObject(with:data) as? [Any])
                    let piece = try XCTUnwrap(old[0] as? String)
                    let score = try XCTUnwrap(old[1] as? NSNumber)
                    let kind = try XCTUnwrap(old[2] as? String)
                    let record = try decoder.decode(Tokenizer.Record.self,from:data)
                    XCTAssertTrue(record.piece.utf8.elementsEqual(piece.utf8))
                    XCTAssertEqual(record.score.bitPattern,score.doubleValue.bitPattern)
                    XCTAssertTrue(record.kind.utf8.elementsEqual(kind.utf8))
                }
            }
        }
        let record = try decoder.decode(Tokenizer.Record.self,from:Data("[\"a\",1,\"NORMAL\",{}]".utf8))
        XCTAssertEqual(record.piece,"a")
        XCTAssertEqual(record.score,1)
        XCTAssertEqual(record.kind,"NORMAL")
    }

    func testMalformedVocabulary() throws {
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:path) }
        for (row,message) in [("[\"a\",1e100,\"NORMAL\"]","Invalid tokenizer score"),
                              ("[\"a\",-1e100,\"NORMAL\"]","Invalid tokenizer score"),
                              ("[\"<0x-1>\",0,\"BYTE\"]","Invalid byte token")] {
            try Data((row+"\n").utf8).write(to:path)
            XCTAssertThrowsError(try Tokenizer(path:path.path)) { error in
                XCTAssertEqual(String(describing:error),message)
            }
        }
    }

    func testMalformedWeights() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let path = directory.appendingPathComponent("bad.safetensors")
        XCTAssertThrowsError(try Auxiliary(path: path.path, device: device))
        for data in [Data(), Data(repeating: 255, count: 16), Data([8,0,0,0,0,0,0,0] + Array("{}      ".utf8))] {
            try data.write(to: path)
            XCTAssertThrowsError(try Weights(path: path.path, device: device))
            XCTAssertThrowsError(try Packed(path: path.path, device: device))
            XCTAssertThrowsError(try Auxiliary(path: path.path, device: device))
        }
    }

    func testAttentionWindowAndRing() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw XCTSkip("Metal unavailable") }
        let source = try String(contentsOf: Bundle.module.url(forResource: "Kernels", withExtension: "metal")!)
        let options = MTLCompileOptions(); options.fastMathEnabled = false
        let library = try device.makeLibrary(source: source, options: options)
        let softmax = try device.makeComputePipelineState(function: library.makeFunction(name: "softmax")!)
        let attend = try device.makeComputePipelineState(function: library.makeFunction(name: "attend")!)
        for iteration in 0..<32 {
            let full = iteration % 2 == 0
            let start = full ? 65 : 4090, base = full ? 0 : 2043
            let count = start+2-base, capacity = full ? count : 2064
            let scores = device.makeBuffer(length: 32*count*4)!
            memset(scores.contents(), 0, scores.length)
            let values = device.makeBuffer(length: capacity*128*2)!
            let v = values.contents().assumingMemoryBound(to: UInt16.self)
            for pos in base..<(start+2) {
                for d in 0..<128 { v[(pos%capacity)*128+d] = UInt16(Float(pos%17-8).bitPattern >> 16) }
            }
            let output = device.makeBuffer(length: 32*128*4)!
            let command = queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
            var a = Args(batch: 2, start: UInt32(start), count: UInt32(count), capacity: UInt32(capacity), base: UInt32(base), full: full ? 1 : 0)
            encoder.setBytes(&a, length: MemoryLayout<Args>.size, index: 3)
            encoder.setBuffer(scores, offset: 0, index: 0)
            encoder.setBuffer(scores, offset: 0, index: 1)
            encoder.setComputePipelineState(softmax)
            encoder.dispatchThreadgroups(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.setComputePipelineState(attend)
            encoder.setBuffer(values, offset: 0, index: 1); encoder.setBuffer(output, offset: 0, index: 2)
            encoder.dispatchThreadgroups(MTLSize(width: 4, height: 2, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
            XCTAssertEqual(command.status, .completed)
            let actual = output.contents().assumingMemoryBound(to: Float.self)
            for token in 0..<2 {
                let begin = full ? 0 : start+token-2047
                let expected = Float((begin...start+token).reduce(0) { $0 + $1%17-8 })/Float(start+token-begin+1)
                for i in 0..<2048 { XCTAssertEqual(actual[token*2048+i], expected, accuracy: 0.0003) }
            }
        }
    }

    func testScoreMemoryFollowsBatch() throws {
        guard FileManager.default.fileExists(atPath: "models/plamo/model.p12") else { throw XCTSkip("Pack weights first") }
        for context in [64,32768] {
            let model = try Runtime(directory: "models/plamo", context: context, packed: true)
            XCTAssertTrue(model.scores.buffer === model.gates.buffer)
            XCTAssertTrue(model.logits.buffer === model.gates.buffer)
            XCTAssertTrue(model.projected.buffer === model.qkv.buffer)
            XCTAssertTrue(model.normalized.buffer === model.activated.buffer)
            _ = try model.prefill([1]+Array(500..<532))
            XCTAssertTrue(model.scores.buffer === model.gates.buffer)
            let expected = model.logitValues().map(\.bitPattern)
            model.reset()
            _ = try model.prefill([1]+Array(500..<532))
            XCTAssertEqual(model.logitValues().map(\.bitPattern),expected)
            try model.forward(Array(600..<616))
            XCTAssertGreaterThanOrEqual(model.scores.buffer.length,16*16*49*4)
        }
    }

    func testSuppliedDevice() throws {
        guard FileManager.default.fileExists(atPath:"models/plamo/model.p12") else { throw XCTSkip("Pack weights first") }
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let model = try Runtime(directory:"models/plamo",context:64,packed:true,device:device)
        XCTAssertTrue(model.device === device)
        _ = try model.prefill([1,500,600])
        XCTAssertTrue(model.logitValues().allSatisfy(\.isFinite))
        XCTAssertThrowsError(try Runtime(directory:"missing-model",context:64,device:nil)) { error in
            XCTAssertTrue(String(describing:error).contains("Apple silicon Metal is required"))
        }
    }

    func testConfiguredKernels() throws {
        guard FileManager.default.fileExists(atPath:"models/plamo/model.safetensors") else { throw XCTSkip("Download weights first") }
        for context in [64,8191,8192] { try checkKernelSet(context:context,budget:nil,mtp:false) }
    }

    func testPackedKernelSet() throws {
        guard FileManager.default.fileExists(atPath:"models/plamo/model.p12") else { throw XCTSkip("Pack weights first") }
        let model = try Runtime(directory:"models/plamo",context:64,packed:true)
        XCTAssertEqual(model.pipelines.count,13)
        XCTAssertNotNil(model.pipelines["matrix_rotary"])
        XCTAssertNotNil(model.pipelines["linear_packed"])
        XCTAssertNotNil(model.pipelines["matrix_packed"])
        XCTAssertNil(model.pipelines["linear_pair_packed"])
        _ = try model.prefill([1]+Array(500..<517))
        try model.forward([600])
        XCTAssertTrue(model.logitValues().allSatisfy(\.isFinite))
    }

    func testAuxiliaryKernels() throws {
        guard FileManager.default.fileExists(atPath:"models/plamo/auxiliary.safetensors") else { throw XCTSkip("Train adapters first") }
        for (context,budget,mtp) in [(64,Optional(16),false),(64,Optional(128),false),(8192,nil,true),(64,Optional(16),true)] {
            try checkKernelSet(context:context,budget:budget,mtp:mtp)
        }
    }

    private func checkKernelSet(context: Int, budget: Int?, mtp: Bool) throws {
        try autoreleasepool {
            let model = try Runtime(directory:"models/plamo",context:context,qsaBudget:budget,mtp:mtp)
            let names = Set(model.pipelines.keys)
            for name in ["norm","linear","matrix","swiglu","qkv","rotate","matrix_rotary","softmax","attend","argmax","init_rope","matrix_raw"] {
                XCTAssertTrue(names.contains(name),name)
            }
            XCTAssertEqual(names.contains("linear_pair"),mtp)
            XCTAssertEqual(names.contains("rotate_steps"),mtp)
            XCTAssertEqual(names.contains("mtp_activate"),mtp)
            XCTAssertEqual(names.contains("mtp_add"),mtp)
            XCTAssertEqual(names.contains("qsa_select"),budget.map { context > $0 } ?? false)
            XCTAssertEqual(names.contains("attention"),context >= 8192 && budget == nil && !mtp)
            XCTAssertEqual(names.contains("attention_merge"),names.contains("attention"))
            if budget == nil && !mtp { XCTAssertEqual(names.count,context >= 8192 ? 14 : 12) }
            _ = try model.prefill([1]+Array(500..<517))
            for token in [600,700] { try model.forward([token]) }
            XCTAssertTrue(model.logitValues().allSatisfy(\.isFinite))
        }
    }

    func testRepeatedPrefillReleasesWork() throws {
        guard FileManager.default.fileExists(atPath: "models/plamo/model.p12") else { throw XCTSkip("Pack weights first") }
        let model = try autoreleasepool { try Runtime(directory:"models/plamo",context:128,packed:true) }
        var allocations = [Int]()
        for _ in 0..<5 {
            model.reset()
            _ = try model.prefill([1]+Array(500..<532))
            for token in [600,700,800] { try model.forward([token]) }
            allocations.append(model.device.currentAllocatedSize)
        }
        XCTAssertLessThanOrEqual(allocations.max()!-allocations.min()!,16384)
        XCTAssertGreaterThanOrEqual(model.peakMetalBytes,allocations.max()!)
        XCTAssertTrue(model.scores.buffer === model.gates.buffer)
    }

    func testFusedNormMatchesSeparate() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw XCTSkip("Metal unavailable") }
        let source = try String(contentsOf: Bundle.module.url(forResource: "Kernels", withExtension: "metal")!)
        let options = MTLCompileOptions(); options.fastMathEnabled = false
        let library = try device.makeLibrary(source: source, options: options)
        let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "norm")!)
        let weights = (0..<3072).map { UInt16((Float($0%31-15)/64).bitPattern >> 16) }
        let w = weights.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!,length: $0.count)! }
        for batch in [1,3,16] {
            let input = (0..<batch*1536).map { Float($0%37-18)/16 }
            let residual = input.map { $0/2 }
            let x = input.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!,length: $0.count)! }
            var expected = [UInt32]()
            for fused in [false,true] {
                let r = residual.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!,length: $0.count)! }
                let y = device.makeBuffer(length: input.count*4)!
                let command = queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
                encoder.setComputePipelineState(pipeline)
                func dispatch(_ source: MTLBuffer, _ target: MTLBuffer, _ weightOffset: Int, _ offset: Float, _ add: Bool, _ next: Bool) {
                    encoder.setBuffer(w,offset: weightOffset,index: 0)
                    encoder.setBuffer(source,offset: 0,index: 1); encoder.setBuffer(target,offset: 0,index: 2)
                    encoder.setBuffer(r,offset: 0,index: 4); encoder.setBuffer(w,offset: 1536*2,index: 5)
                    encoder.setBuffer(y,offset: 0,index: 6)
                    var a = Args(n: 1536,m: next ? 1 : 0,batch: UInt32(batch),full: add ? 1 : 0,offset: offset)
                    encoder.setBytes(&a,length: MemoryLayout<Args>.size,index: 3)
                    encoder.dispatchThreadgroups(MTLSize(width: batch,height: 1,depth: 1),threadsPerThreadgroup: MTLSize(width: 256,height: 1,depth: 1))
                }
                dispatch(x,r,0,0.2,true,fused)
                if !fused { dispatch(r,y,1536*2,1,false,false) }
                encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                XCTAssertEqual(command.status,.completed)
                let actual = [r,y].flatMap { buffer in
                    Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: UInt32.self),count: input.count))
                }
                if expected.isEmpty { expected = actual } else { XCTAssertEqual(actual,expected) }
            }
        }
    }

    func testMetalLinearAndMatrix() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw XCTSkip("Metal unavailable") }
        let source = try String(contentsOf: Bundle.module.url(forResource: "Kernels", withExtension: "metal")!)
        let options = MTLCompileOptions(); options.fastMathEnabled = false
        let library = try device.makeLibrary(source: source, options: options)
        func bf(_ x: Float) -> Float {
            let u = x.bitPattern
            return Float(bitPattern: ((u &+ 0x7fff &+ ((u >> 16) & 1)) >> 16) << 16)
        }
        for batch in [1, 3, 16, 32, 33, 64] {
            let n = 128, m = 35
            let w = (0..<n*m).map { UInt16(bf(Float($0%17-8)/16).bitPattern >> 16) }
            let x = (0..<n*batch).map { bf(Float($0%11-5)/8) }
            let wb = w.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)! }
            let xb = x.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)! }
            let yb = device.makeBuffer(length: m*batch*4)!
            let name = batch == 1 ? "linear" : "matrix"
            let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name:name,constantValues:MTLFunctionConstantValues()))
            let command = queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(wb, offset: 0, index: 0); encoder.setBuffer(xb, offset: 0, index: 1); encoder.setBuffer(yb, offset: 0, index: 2)
            encoder.setBuffer(wb, offset: 0, index: 4); encoder.setBuffer(wb, offset: 0, index: 5)
            var args = Args(n: UInt32(n), m: UInt32(m), batch: UInt32(batch))
            encoder.setBytes(&args, length: MemoryLayout<Args>.size, index: 3)
            let columns = 16
            encoder.dispatchThreadgroups(MTLSize(width: batch == 1 ? (m*32+255)/256 : (m+31)/32, height: batch == 1 ? 1 : (batch+columns-1)/columns, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
            XCTAssertEqual(command.status, .completed, "\(String(describing: command.error))")
            let output = yb.contents().assumingMemoryBound(to: Float.self)
            for t in 0..<batch {
                for r in 0..<m {
                    var sum: Float = 0
                    for k in 0..<n { sum += Float(bitPattern: UInt32(w[r*n+k]) << 16)*x[t*n+k] }
                    XCTAssertEqual(output[t*m+r], bf(sum), accuracy: 0.0001)
                }
            }
        }
    }
}
