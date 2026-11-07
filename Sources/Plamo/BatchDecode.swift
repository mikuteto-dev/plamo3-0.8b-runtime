import Foundation

struct BatchOutput {
    let ids: [Int]
    let reason: String
}

func decodeBatch(_ initial: [Int], limits: [Int], step: ([Int?]) throws -> [Int],
                 emit: (Int,Int) throws -> Void = { _,_ in },
                 finish: (Int,BatchOutput) throws -> Void = { _,_ in }, stopOnTruncation: Bool = false) throws -> [BatchOutput] {
    let count = initial.count
    guard (1...4).contains(count), initial.allSatisfy({ (0..<107520).contains($0) }),
          limits.count == count, limits.allSatisfy({ (0...8191).contains($0) }) else { throw Failure.message("Invalid quad decode inputs") }
    var next = initial, outputs = [[Int]](repeating:[],count:count), reasons = [String?](repeating:nil,count:count)
    var last = count-1
    while reasons.prefix(last+1).contains(where:{ $0 == nil }) {
        var tokens = [Int?](repeating:nil,count:count)
        for lane in 0..<count {
            if lane > last { break }
            if reasons[lane] != nil { continue }
            if outputs[lane].count == limits[lane] { reasons[lane] = "length" }
            else if next[lane] == 1 || next[lane] == 16 { reasons[lane] = "stop" }
            else {
                outputs[lane].append(next[lane]); try emit(lane,next[lane])
                if outputs[lane].count == limits[lane] { reasons[lane] = "length" }
                else { tokens[lane] = next[lane] }
            }
            if let reason = reasons[lane] {
                if stopOnTruncation && reason == "length" { last = min(last,lane) }
                try finish(lane,BatchOutput(ids:outputs[lane],reason:reason))
            }
        }
        if tokens.allSatisfy({ $0 == nil }) { break }
        next = try step(tokens)
        guard next.count == count, next.allSatisfy({ (0..<107520).contains($0) }) else { throw Failure.message("Invalid quad predictions") }
    }
    return (0...last).map { BatchOutput(ids:outputs[$0],reason:reasons[$0]!) }
}

extension Batch {
    func generate(_ prompts: [[Int]], limits: [Int], emit: (Int,Int) throws -> Void = { _,_ in },
                  finish: (Int,BatchOutput) throws -> Void = { _,_ in }, stopOnTruncation: Bool = false) throws -> [BatchOutput] {
        guard (1...4).contains(prompts.count), limits.count == prompts.count, limits.allSatisfy({ (0...8191).contains($0) }),
              zip(prompts,limits).allSatisfy({ $0.0.count+$0.1 <= scalar.context }) else { throw Failure.message("Quad output budget exceeds context") }
        do { return try decodeBatch(prefill(prompts),limits:limits,step:{ try self.step($0) },emit:emit,finish:finish,stopOnTruncation:stopOnTruncation) }
        catch { reset(); throw error }
    }
}
