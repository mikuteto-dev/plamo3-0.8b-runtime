import XCTest
import CryptoKit
@testable import Plamo

final class CacheTests: XCTestCase {
    private func fixture() throws -> (URL,Data,Data) {
        let model = "models/plamo/tokenizer.jsonl"
        guard FileManager.default.fileExists(atPath:model) else { throw XCTSkip("Download vocabulary first") }
        let source = try Data(contentsOf:URL(fileURLWithPath:model))
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        let path = directory.appendingPathComponent("tokenizer.jsonl")
        try source.write(to:path)
        let plain = try Tokenizer(path:path.path)
        let fields = Dictionary(uniqueKeysWithValues:Mirror(reflecting:plain).children.map { ($0.label!,$0.value) })
        let nodes = try XCTUnwrap(fields["nodes"] as? [Tokenizer.Node])
        let scores = try XCTUnwrap(fields["scores"] as? [Int64])
        let roots = try XCTUnwrap(fields["roots"] as? [UInt32:Int])
        let bytes = try XCTUnwrap(fields["bytes"] as? [Int])
        var cache = Data([80,76,65,77,79,1,0,0]), words = [UInt32]()
        words.reserveCapacity(nodes.count*4)
        for node in nodes {
            words.append(UInt32(bitPattern:node.first).littleEndian)
            words.append(UInt32(bitPattern:node.token).littleEndian)
            words.append(node.scalar.littleEndian)
            words.append(UInt32(bitPattern:node.next).littleEndian)
        }
        words.withUnsafeBytes { cache.append(contentsOf:$0) }
        scores.map(\.littleEndian).withUnsafeBytes { cache.append(contentsOf:$0) }
        words = roots.keys.sorted().flatMap { [$0.littleEndian,UInt32(roots[$0]!).littleEndian] }
        words.withUnsafeBytes { cache.append(contentsOf:$0) }
        bytes.map { UInt32($0).littleEndian }.withUnsafeBytes { cache.append(contentsOf:$0) }
        var offsets: [UInt32] = [0], text = Data()
        for piece in plain.pieces { text.append(contentsOf:piece.utf8);offsets.append(UInt32(text.count).littleEndian) }
        offsets.withUnsafeBytes { cache.append(contentsOf:$0) };cache.append(text)
        XCTAssertEqual(cache.count,5337561)
        XCTAssertEqual(SHA256.hash(data:cache).map { String(format:"%02x",$0) }.joined(),
                       "325c68005301172dffa3124a5ef813537d0e666a9b2c6411c4f406fd16511e53")
        return (path,source,cache)
    }

    func testCacheMatchesSource() throws {
        let (path,source,cache) = try fixture()
        defer { try? FileManager.default.removeItem(at:path.deletingLastPathComponent()) }
        let plain = try Tokenizer(path:path.path)
        let cachePath = URL(fileURLWithPath:path.path+".trie")
        try cache.write(to:cachePath)
        try FileManager.default.setAttributes([.posixPermissions:0o444],ofItemAtPath:path.path)
        try FileManager.default.setAttributes([.posixPermissions:0o444],ofItemAtPath:cachePath.path)
        let cached = try Tokenizer(path:path.path)
        XCTAssertEqual(plain.pieces,cached.pieces)
        for (a,b) in zip(plain.pieces,cached.pieces) { XCTAssertTrue(a.utf8.elementsEqual(b.utf8)) }
        let old = Dictionary(uniqueKeysWithValues:Mirror(reflecting:plain).children.map { ($0.label!,$0.value) })
        let new = Dictionary(uniqueKeysWithValues:Mirror(reflecting:cached).children.map { ($0.label!,$0.value) })
        let before = try XCTUnwrap(old["nodes"] as? [Tokenizer.Node])
        let after = try XCTUnwrap(new["nodes"] as? [Tokenizer.Node])
        XCTAssertEqual(before.count,after.count)
        for (a,b) in zip(before,after) {
            XCTAssertEqual(a.first,b.first);XCTAssertEqual(a.next,b.next)
            XCTAssertEqual(a.scalar,b.scalar);XCTAssertEqual(a.token,b.token)
        }
        XCTAssertEqual(try XCTUnwrap(old["maxSpan"] as? Int),32)
        XCTAssertEqual(try XCTUnwrap(new["maxSpan"] as? Int),32)
        XCTAssertEqual(old["scores"] as? [Int64],new["scores"] as? [Int64])
        XCTAssertEqual(old["roots"] as? [UInt32:Int],new["roots"] as? [UInt32:Int])
        XCTAssertEqual(old["bytes"] as? [Int],new["bytes"] as? [Int])
        for text in ["", "a aba", "会議は月曜日です。", "🧪𠮷 e\u{301}", "<|plamo:bos|>"] {
            XCTAssertEqual(plain.encode(text),cached.encode(text))
        }
        XCTAssertEqual(try Data(contentsOf:path),source)
    }

    func testPackedDecodeMatchesSource() throws {
        let (path,source,cache) = try fixture()
        defer { try? FileManager.default.removeItem(at:path.deletingLastPathComponent()) }
        let plain = try Tokenizer(path:path.path)
        try cache.write(to:URL(fileURLWithPath:path.path+".trie"))
        let cached = try Tokenizer(path:path.path)
        for id in 0..<107520 {
            XCTAssertTrue(try plain.decode([id]).utf8.elementsEqual(cached.decode([id]).utf8),"id=\(id)")
        }
        let fields = Dictionary(uniqueKeysWithValues:Mirror(reflecting:plain).children.map { ($0.label!,$0.value) })
        let bytes = try XCTUnwrap(fields["bytes"] as? [Int])
        for value: [UInt8] in [[0], [255], [192,128], [237,160,128], [226,130], [240,159,167,170], [195,169], Array(0...255)] {
            let ids = value.map { bytes[Int($0)] }
            XCTAssertTrue(try cached.decode(ids).utf8.elementsEqual(String(decoding:value,as:UTF8.self).utf8))
            XCTAssertTrue(try plain.decode(ids).utf8.elementsEqual(cached.decode(ids).utf8))
            XCTAssertTrue(try plain.decode([97]+ids+[98]).utf8.elementsEqual(cached.decode([97]+ids+[98]).utf8))
        }
        for ids in [[-1], [107520], [97,-1], [97,107520], [128,Int.max]] {
            XCTAssertThrowsError(try cached.decode(ids))
        }
        XCTAssertEqual(try cached.decode([]),"")
        var lines = String(decoding:source,as:UTF8.self).split(separator:"\n").map(String.init)
        lines.swapAt(128,384)
        try lines.joined(separator:"\n").write(to:path,atomically:true,encoding:.utf8)
        let custom = try Tokenizer(path:path.path)
        XCTAssertEqual(try custom.decode([128]),"\u{0}")
        XCTAssertEqual(try custom.decode([384]),"\u{fffd}")
        XCTAssertEqual(try cached.decode([128]),"\u{fffd}")
        XCTAssertEqual(try cached.decode([384]),"\u{0}")
    }

    func testLoadedPiecesOwnTheirBytes() throws {
        let (path,_,cache) = try fixture()
        defer { try? FileManager.default.removeItem(at:path.deletingLastPathComponent()) }
        let plain = try Tokenizer(path:path.path)
        let cachePath = URL(fileURLWithPath:path.path+".trie")
        try cache.write(to:cachePath)
        let cached = try Tokenizer(path:path.path)
        let ids = Array(0..<107520)
        let expected = try plain.decode(ids)
        let text = "会議は月曜日です。 🧪 e\u{301}"
        let tokens = plain.encode(text)
        try Data().write(to:cachePath)
        try Data().write(to:path)
        XCTAssertEqual(cached.pieces,plain.pieces)
        XCTAssertEqual(try cached.decode(ids),expected)
        XCTAssertEqual(cached.encode(text),tokens)
        XCTAssertEqual(try cached.decode([]),"")
        for invalid in [-1,107520] { XCTAssertThrowsError(try cached.decode([invalid])) }
        XCTAssertThrowsError(try Tokenizer(path:path.path))
    }

    func testInvalidCacheAndChangedSource() throws {
        let (path,source,cache) = try fixture()
        defer { try? FileManager.default.removeItem(at:path.deletingLastPathComponent()) }
        let plain = try Tokenizer(path:path.path), cachePath = URL(fileURLWithPath:path.path+".trie")
        func check(_ bad: Data) throws {
            try bad.write(to:cachePath)
            XCTAssertEqual(try Tokenizer(path:path.path).encode("aba"),plain.encode("aba"))
        }
        for bad in [Data(),Data(cache.dropLast()),cache+Data([0])] { try check(bad) }
        for index in [0,5,8,3129992,3990152,4045136,4046160,4476244,cache.count-1] {
            var corrupt = cache;corrupt[index] ^= 1
            try check(corrupt)
        }
        try cache.write(to:cachePath)
        var changed = source
        let old = Data("[\"a\", -7.4026, \"NORMAL\"]".utf8)
        let range = try XCTUnwrap(changed.range(of:old))
        changed.replaceSubrange(range,with:Data("[\"a\", 99.9999, \"NORMAL\"]".utf8))
        XCTAssertEqual(changed.count,source.count)
        try changed.write(to:path)
        let custom = try Tokenizer(path:path.path)
        let a = try XCTUnwrap(custom.pieces.firstIndex(of:"a")), b = try XCTUnwrap(custom.pieces.firstIndex(of:"b"))
        XCTAssertEqual(custom.encode("aba"),[a,b,a])
        XCTAssertNotEqual(custom.encode("aba"),plain.encode("aba"))
        XCTAssertEqual(try Data(contentsOf:path),changed)
        try Data("[\"a\",1e100,\"NORMAL\"]\n".utf8).write(to:path)
        XCTAssertThrowsError(try Tokenizer(path:path.path)) { error in
            XCTAssertEqual(String(describing:error),"Invalid tokenizer score")
        }
        try FileManager.default.removeItem(at:path)
        XCTAssertThrowsError(try Tokenizer(path:path.path))
    }
}
