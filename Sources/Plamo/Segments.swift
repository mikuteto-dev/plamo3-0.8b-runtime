import Foundation
import NaturalLanguage

struct Segment {
    var text: String
    let tokens: [Int]

    static func split(_ text: String, target: String, limit: Int, tokenizer: Tokenizer) throws -> [Segment] {
        _ = try Tokenizer.prompt(text, target: target)
        guard limit > 0 else { throw Failure.message("No context space for a translation segment") }
        var parts = [Segment]()
        func literal(_ text: String) {
            guard !text.isEmpty else { return }
            if parts.last?.tokens.isEmpty == true {
                parts[parts.count-1].text += text
            } else {
                parts.append(Segment(text:text,tokens:[]))
            }
        }
        func line(_ text: String) throws {
            if text.allSatisfy({ $0.isWhitespace }) { literal(text); return }
            let characters = Array(text)
            let sentences = NLTokenizer(unit:.sentence)
            sentences.string = text
            var ends = [Int]()
            sentences.enumerateTokens(in:text.startIndex..<text.endIndex) { range, _ in
                ends.append(text.distance(from:text.startIndex,to:range.upperBound))
                return true
            }
            var start = 0
            while start < characters.count {
                let first = start
                while start < characters.count && characters[start].isWhitespace { start += 1 }
                literal(String(characters[first..<start]))
                if start == characters.count { break }
                var upper = min(characters.count,start+limit*4)
                while true {
                    let sentenceEnd = ends.last(where:{ $0 > start && $0 <= upper })
                    var end = sentenceEnd ?? upper
                    if sentenceEnd == nil && upper < characters.count,
                       let space = (start..<upper).last(where:{ characters[$0].isWhitespace }) {
                        end = space
                    }
                    while end > start && characters[end-1].isWhitespace { end -= 1 }
                    let source = String(characters[start..<end])
                    let tokens = tokenizer.encode(try Tokenizer.prompt(source,target:target),bos:true)
                    if tokens.count <= limit {
                        parts.append(Segment(text:source,tokens:tokens))
                        start = end
                        break
                    }
                    guard end-start > 1 else { throw Failure.message("A single character exceeds the segment token limit") }
                    upper = start+(end-start)/2
                }
            }
        }
        var pending = ""
        for character in text {
            if character.isNewline {
                try line(pending)
                pending = ""
                literal(String(character))
            } else {
                pending.append(character)
            }
        }
        try line(pending)
        guard parts.contains(where:{ !$0.tokens.isEmpty }) else { throw Failure.message("Source must contain text") }
        return parts
    }
}
