import XCTest
@testable import Plamo

final class CostTests: XCTestCase {
    private func fullCosts(_ text: String, _ tokenizer: Tokenizer, bos: Bool = false) throws -> [Int] {
        let fields=Dictionary(uniqueKeysWithValues:Mirror(reflecting:tokenizer).children.map { ($0.label!,$0.value) })
        let nodes=try XCTUnwrap(fields["nodes"] as? [Tokenizer.Node])
        let scores=try XCTUnwrap(fields["scores"] as? [Int64])
        let roots=try XCTUnwrap(fields["roots"] as? [UInt32:Int])
        let bytes=try XCTUnwrap(fields["bytes"] as? [Int])
        let boundaries=try XCTUnwrap(fields["boundaries"] as? NSRegularExpression)
        let marked=boundaries.stringByReplacingMatches(in:text,range:NSRange(text.startIndex...,in:text),withTemplate:"\u{ee00}$0\u{ee00}")
        var result=bos ? [1] : []
        for segment in marked.unicodeScalars.split(whereSeparator:{ $0.value == 0xee00 }) {
            let scalars=Array(segment),n=scalars.count
            // Keep the original full-size recurrence as an independent windowing reference.
            var costs=[Int64](repeating:Int64.max/4,count:n+1)
            var lengths=[Int](repeating:1,count:n),tokens=[Int](repeating:-1,count:n)
            costs[n]=0
            for i in stride(from:n-1,through:0,by:-1) {
                costs[i]=costs[i+1]+10_000_000
                var node=0
                for j in i..<n {
                    let next: Int?
                    if node == 0 { next=roots[scalars[j].value] }
                    else {
                        var edge=nodes[node].first
                        while edge>=0 && nodes[Int(edge)].scalar != scalars[j].value { edge=nodes[Int(edge)].next }
                        next=edge>=0 ? Int(edge) : nil
                    }
                    guard let next else { break }
                    node=next
                    let token=nodes[node].token
                    if token>=0 {
                        let cost=costs[j+1]-scores[Int(token)]
                        if cost<=costs[i] { costs[i]=cost;lengths[i]=j-i+1;tokens[i]=Int(token) }
                    }
                }
            }
            var i=0
            while i<n {
                if tokens[i]>=0 { result.append(tokens[i]) }
                else { result.append(contentsOf:String(scalars[i]).utf8.map { bytes[Int($0)] }) }
                i+=lengths[i]
            }
        }
        return result
    }

    func testWindowBoundariesMatchFullCosts() throws {
        let path="models/plamo/tokenizer.jsonl"
        guard FileManager.default.fileExists(atPath:path) else { throw XCTSkip("Download vocabulary first") }
        let tokenizer=try Tokenizer(path:path)
        let unicode=Array("🧪𠮷e\u{301}日 a\t".unicodeScalars)
        for n in [0,1,2,31,32,33,62,63,64,65,95,96,97,127,128,129,255,256,257,1025] {
            var text=""
            for i in 0..<n { text.unicodeScalars.append(unicode[i%unicode.count]) }
            for value in [text,String(repeating:" ",count:n),String(repeating:"ab",count:n)] {
                XCTAssertEqual(tokenizer.encode(value),try fullCosts(value,tokenizer),"n=\(n)")
                XCTAssertEqual(tokenizer.encode(value,bos:true),try fullCosts(value,tokenizer,bos:true),"BOS n=\(n)")
            }
        }
        let marked="<|plamo:tag|>input<|plamo:msg|>"+String(repeating:"日本語 ab ",count:50)+"aaaaaa\n\n\n\n"
        XCTAssertEqual(tokenizer.encode(marked),try fullCosts(marked,tokenizer))
    }

    func testModifiedScoresAcrossWraps() throws {
        let source="models/plamo/tokenizer.jsonl"
        guard FileManager.default.fileExists(atPath:source) else { throw XCTSkip("Download vocabulary first") }
        let pieces=try Tokenizer(path:source).pieces
        let a=try XCTUnwrap(pieces.firstIndex(of:"a")),b=try XCTUnwrap(pieces.firstIndex(of:"b")),ab=try XCTUnwrap(pieces.firstIndex(of:"ab"))
        let spaces=try XCTUnwrap(pieces.firstIndex(of:String(repeating:" ",count:32)))
        var lines=try String(contentsOfFile:source,encoding:.utf8).split(separator:"\n").map(String.init)
        lines[a]="[\"a\",300000,\"NORMAL\"]";lines[b]="[\"b\",300000,\"NORMAL\"]"
        let record=try JSONSerialization.data(withJSONObject:[String(repeating:" ",count:32),100000,"NORMAL"])
        lines[spaces]=String(decoding:record,as:UTF8.self)
        let path=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:path) }
        for score in [500000,600000,700000] {
            lines[ab]="[\"ab\",\(score),\"NORMAL\"]"
            try lines.joined(separator:"\n").write(to:path,atomically:true,encoding:.utf8)
            let tokenizer=try Tokenizer(path:path.path)
            for text in [String(repeating:"ab",count:129)+"a",String(repeating:" ",count:95),String(repeating:" ",count:96),String(repeating:" ",count:97)] {
                XCTAssertEqual(tokenizer.encode(text),try fullCosts(text,tokenizer),"score=\(score)")
            }
        }
    }
}
