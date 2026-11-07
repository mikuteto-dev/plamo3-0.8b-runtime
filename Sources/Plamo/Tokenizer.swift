import Foundation
import CryptoKit

enum Failure: Error, CustomStringConvertible {
    case message(String)
    var description: String {
        switch self { case .message(let message): return message }
    }
}

struct Tokenizer {
    struct Record: Decodable {
        let piece: String
        let score: Double
        let kind: String

        init(from decoder: Decoder) throws {
            var fields = try decoder.unkeyedContainer()
            piece = try fields.decode(String.self)
            score = try fields.decode(Double.self)
            kind = try fields.decode(String.self)
        }
    }

    struct Node {
        var first: Int32 = -1
        var token: Int32 = -1
        var scalar: UInt32 = 0
        var next: Int32 = -1
    }
    private var nodes = [Node()]
    private var scores = [Int64]()
    private var maxSpan = 1
    private var roots = [UInt32: Int]()
    private var strings = [String]()
    private var pieceData: Data?
    private var count: Int { pieceData == nil ? strings.count : 107520 }
    // Keep full-vocabulary diagnostics without retaining their decoded strings.
    var pieces: [String] { pieceData == nil ? strings : (0..<count).map { piece($0) } }
    private var bytes = [Int](repeating: -1, count: 256)
    private let boundaries = try! NSRegularExpression(
        pattern: #"(<\|plamo:[^|\s\u001c-\u001f]{0,64}\|>)|(([^\n])\3{3,}| {2,})"#
    )

    private static func cache(_ path: String) -> Data? {
        let cachePath = path+".trie"
        guard let attributes = try? FileManager.default.attributesOfItem(atPath:cachePath),
              attributes[.size] as? Int == 5337561,
              let source = try? Data(contentsOf:URL(fileURLWithPath:path),options:.mappedIfSafe),
              source.count == 3526723 else { return nil }
        func sha(_ data: Data) -> String { SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined() }
        guard sha(source) == "bdce9e430e65931b28c56de2054f52b4cf8ed59b322db125e016d724d06dbb0e",
              let data = try? Data(contentsOf:URL(fileURLWithPath:cachePath),options:.mappedIfSafe),
              data.count == 5337561, data.prefix(8).elementsEqual([80,76,65,77,79,1,0,0]),
              sha(data) == "325c68005301172dffa3124a5ef813537d0e666a9b2c6411c4f406fd16511e53" else { return nil }
        return data
    }

    init(path: String) throws {
        if let data = Self.cache(path) {
            // The validated vocabulary SHA pins its longest token at 32 scalars.
            maxSpan = 32
            // Fixed offsets are used only after validating the entire model-specific cache.
            data.withUnsafeBytes { p in
                nodes = (0..<195624).map { i in
                    let offset = 8+i*16
                    return Node(first:p.loadUnaligned(fromByteOffset:offset,as:Int32.self).littleEndian,
                                token:p.loadUnaligned(fromByteOffset:offset+4,as:Int32.self).littleEndian,
                                scalar:p.loadUnaligned(fromByteOffset:offset+8,as:UInt32.self).littleEndian,
                                next:p.loadUnaligned(fromByteOffset:offset+12,as:Int32.self).littleEndian)
                }
                scores = (0..<107520).map { p.loadUnaligned(fromByteOffset:3129992+$0*8,as:Int64.self).littleEndian }
                bytes = (0..<256).map { Int(p.loadUnaligned(fromByteOffset:4045136+$0*4,as:UInt32.self).littleEndian) }
                roots.reserveCapacity(6873)
                for i in 0..<6873 {
                    roots[p.loadUnaligned(fromByteOffset:3990152+i*8,as:UInt32.self).littleEndian] = Int(p.loadUnaligned(fromByteOffset:3990156+i*8,as:UInt32.self).littleEndian)
                }
                // Own only the packed strings so later cache edits cannot alter loaded vocabulary.
                pieceData = Data(bytes:p.baseAddress!.advanced(by:4046160),count:p.count-4046160)
            }
        } else {
            let contents = try String(contentsOfFile:path,encoding:.utf8)
            strings.reserveCapacity(107520)
            scores.reserveCapacity(107520)
            let decoder = JSONDecoder()
            for line in contents.split(separator:"\n") {
                try append(decoder.decode(Record.self,from:Data(line.utf8)))
            }
        }
        guard count == 107520, bytes.allSatisfy({ $0 >= 0 }),
              piece(1) == "<|plamo:bos|>", piece(16) == "<|plamo:tag|>" else {
            throw Failure.message("Expected the PLaMo 3 Translate 0.8B vocabulary")
        }
    }

    private mutating func append(_ record: Record) throws {
        let piece = record.piece
        let id = strings.count
        strings.append(record.piece)
        if record.kind == "BYTE" {
            guard piece.count == 6, piece.hasPrefix("<0x"), piece.hasSuffix(">"),
                  let byte = Int(piece.dropFirst(3).dropLast(), radix: 16), (0..<256).contains(byte) else {
                throw Failure.message("Invalid byte token")
            }
            bytes[byte] = id
            scores.append(0)
            return
        }
        let scaled = (record.score*1e4).rounded(.toNearestOrEven)
        guard scaled.isFinite, scaled >= Double(Int64.min), scaled < Double(Int64.max) else {
            throw Failure.message("Invalid tokenizer score")
        }
        scores.append(Int64(scaled))
        maxSpan = max(maxSpan,piece.unicodeScalars.count)
        var node = 0
        for scalar in piece.unicodeScalars {
            if let child = child(node, scalar.value) {
                node = child
            } else {
                let next = nodes.count
                nodes.append(Node(scalar:scalar.value,next:node == 0 ? -1 : nodes[node].first))
                if node == 0 {
                    roots[scalar.value] = next
                } else {
                    nodes[node].first = Int32(next)
                }
                node = next
            }
        }
        nodes[node].token = Int32(id)
    }

    private func child(_ node: Int, _ scalar: UInt32) -> Int? {
        if node == 0 { return roots[scalar] }
        var edge = nodes[node].first
        while edge >= 0 {
            let entry = nodes[Int(edge)]
            if entry.scalar == scalar { return Int(edge) }
            edge = entry.next
        }
        return nil
    }

    func encode(_ text: String, bos: Bool = false) -> [Int] {
        let range = NSRange(text.startIndex..., in: text)
        let marked = boundaries.stringByReplacingMatches(
            in: text, range: range, withTemplate: "\u{ee00}$0\u{ee00}"
        )
        var result = bos ? [1] : []
        for segment in marked.unicodeScalars.split(whereSeparator: { $0.value == 0xee00 }) {
            let scalars = Array(segment)
            let n = scalars.count
            let capacity = min(n+1,1 << (Int.bitWidth-maxSpan.leadingZeroBitCount))
            // All future reads are within maxSpan; -1 keeps short inputs unwrapped.
            let mask = capacity < n+1 ? capacity-1 : -1
            var costs = [Int64](repeating: Int64.max / 4, count: capacity)
            // Trie links are Int32, so matched token spans also fit Int32.
            var lengths = [Int32](repeating: 1, count: n)
            var tokens = [Int32](repeating: -1, count: n)
            costs[n & mask] = 0
            for i in stride(from: n - 1, through: 0, by: -1) {
                let slot = i & mask
                costs[slot] = costs[(i+1) & mask] + 10_000_000
                var node = 0
                for j in i..<n {
                    guard let next = child(node, scalars[j].value) else { break }
                    node = next
                    let entry = nodes[node]
                    if entry.token >= 0 {
                        let cost = costs[(j+1) & mask] - scores[Int(entry.token)]
                        if cost <= costs[slot] {
                            costs[slot] = cost
                            lengths[i] = Int32(j - i + 1)
                            tokens[i] = entry.token
                        }
                    }
                }
            }
            var i = 0
            while i < n {
                if tokens[i] >= 0 {
                    result.append(Int(tokens[i]))
                } else {
                    result.append(contentsOf: String(scalars[i]).utf8.map { bytes[Int($0)] })
                }
                i += Int(lengths[i])
            }
        }
        return result
    }

    private func piece(_ id: Int) -> String {
        guard let data = pieceData else { return strings[id] }
        return data.withUnsafeBytes { p in
            let first = 430084+Int(p.loadUnaligned(fromByteOffset:id*4,as:UInt32.self).littleEndian)
            let last = 430084+Int(p.loadUnaligned(fromByteOffset:(id+1)*4,as:UInt32.self).littleEndian)
            return String(decoding:UnsafeRawBufferPointer(rebasing:p[first..<last]),as:UTF8.self)
        }
    }

    // Appends the raw piece bytes for one token so a full decode and an incremental
    // stream share exactly the same byte-level vocabulary handling.
    func writeBytes(of id: Int, to bytes: inout [UInt8]) throws {
        guard (0..<count).contains(id) else { throw Failure.message("Token ID out of range") }
        if let packed = pieceData {
            packed.withUnsafeBytes { p in
                // BYTE IDs are fixed by the validated vocabulary and cache SHA.
                if (128..<256).contains(id) || (384..<512).contains(id) {
                    bytes.append(UInt8(id < 256 ? id : id-384))
                } else {
                    let first = 430084+Int(p.loadUnaligned(fromByteOffset:id*4,as:UInt32.self).littleEndian)
                    let last = 430084+Int(p.loadUnaligned(fromByteOffset:(id+1)*4,as:UInt32.self).littleEndian)
                    bytes.append(contentsOf:UnsafeRawBufferPointer(rebasing:p[first..<last]))
                }
            }
        } else {
            let piece = strings[id]
            if piece.count == 6, piece.hasPrefix("<0x"), piece.hasSuffix(">"),
               let byte = UInt8(piece.dropFirst(3).dropLast(), radix: 16) {
                bytes.append(byte)
            } else {
                bytes.append(contentsOf: piece.utf8)
            }
        }
    }

    func decode(_ ids: [Int]) throws -> String {
        var bytes = [UInt8]()
        bytes.reserveCapacity(ids.count * 3)
        for id in ids { try writeBytes(of: id, to: &bytes) }
        return String(decoding: bytes, as: UTF8.self)
    }

    // Incrementally decodes generated tokens, holding back an incomplete trailing UTF-8
    // sequence until the remaining bytes arrive. Concatenated output equals decode(ids).
    final class Stream {
        private let tokenizer: Tokenizer
        private var pending = [UInt8]()

        init(_ tokenizer: Tokenizer) { self.tokenizer = tokenizer }

        func push(_ id: Int) throws -> String {
            try tokenizer.writeBytes(of: id, to: &pending)
            let complete = pending.count - Self.incompleteSuffix(pending)
            guard complete > 0 else { return "" }
            let text = String(decoding: pending[0..<complete], as: UTF8.self)
            pending.removeFirst(complete)
            return text
        }

        func finish() -> String {
            defer { pending.removeAll() }
            return String(decoding: pending, as: UTF8.self)
        }

        // Number of trailing bytes (0...3) that are a valid but unfinished UTF-8 sequence.
        private static func incompleteSuffix(_ bytes: [UInt8]) -> Int {
            var trailer = 0
            var i = bytes.count
            while i > 0 && trailer < 3 {
                i -= 1
                let byte = bytes[i]
                if byte & 0xC0 == 0x80 { trailer += 1; continue }
                if byte < 0x80 { return 0 }
                let needed = byte >= 0xF0 ? 4 : byte >= 0xE0 ? 3 : 2
                return bytes.count - i < needed ? bytes.count - i : 0
            }
            return 0
        }
    }

    static func prompt(_ text: String, target: String) throws -> String {
        let languages = Set("ar az be bg bn ca ceb de en es et eu fa fr gu he hi hr hu hy id it ja jv ka kk km ko lt lv ml mr ms my ne nl pa pl ru si sk sl sr sw ta te th tl ur uz vi zh-hans zh-hant".split(separator: " ").map(String.init))
        guard languages.contains(target) else { throw Failure.message("Unsupported target language") }
        guard !text.isEmpty, !text.contains("<|plamo:"), !text.unicodeScalars.contains(where: { $0.value == 0xee00 }) else {
            throw Failure.message("Source must be nonempty and contain no reserved PLaMo delimiters")
        }
        return "<|plamo:tag|>dataset<|plamo:msg|>translation"
            + "<|plamo:tag|>input<|plamo:msg|>" + text
            + "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>" + target + "<|plamo:msg|>"
    }
}
