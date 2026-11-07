import Foundation

struct DocumentPlan {
    let inputs: [[Int]]
    let prefixes: [String]
    let tail: String

    init(_ parts: [Segment]) throws {
        var inputs = [[Int]](), prefixes = [String](), pending = ""
        for part in parts {
            if part.tokens.isEmpty { pending += part.text }
            else {
                prefixes.append(!inputs.isEmpty && pending.isEmpty ? "\n" : pending)
                inputs.append(part.tokens)
                pending = ""
            }
        }
        guard !inputs.isEmpty else { throw Failure.message("No documents to translate") }
        self.inputs = inputs; self.prefixes = prefixes; tail = pending
    }
}

final class OrderedOutput {
    private let prefixes: [String], tail: String
    private let streams: [Tokenizer.Stream]
    private let emit: (String) throws -> Void
    private var buffers: [String], reasons: [String?]
    private var first = 0
    private var closed = false

    init(tokenizer: Tokenizer, prefixes: [String], tail: String, emit: @escaping (String) throws -> Void) throws {
        guard (1...4).contains(prefixes.count) else { throw Failure.message("Expected one to four output prefixes") }
        self.prefixes = prefixes; self.tail = tail; self.emit = emit
        streams = prefixes.map { _ in Tokenizer.Stream(tokenizer) }
        buffers = [String](repeating:"",count:prefixes.count)
        reasons = [String?](repeating:nil,count:prefixes.count)
        if !prefixes[0].isEmpty { try emit(prefixes[0]) }
    }

    func append(_ lane: Int, _ id: Int) throws {
        guard reasons.indices.contains(lane), reasons[lane] == nil, !closed else { throw Failure.message("Output after document finish") }
        try push(lane,streams[lane].push(id))
    }

    private func push(_ lane: Int, _ text: String) throws {
        guard !text.isEmpty else { return }
        if lane == first { try emit(text) }
        else { buffers[lane] += text }
    }

    func finish(_ lane: Int, _ output: BatchOutput) throws {
        guard reasons.indices.contains(lane), reasons[lane] == nil, !closed,
              output.reason == "stop" || output.reason == "length" else { throw Failure.message("Invalid document finish") }
        try push(lane,streams[lane].finish())
        reasons[lane] = output.reason
        while first < reasons.count, let reason = reasons[first] {
            if reason != "stop" { closed = true; return }
            first += 1
            if first == reasons.count {
                closed = true
                if !tail.isEmpty { try emit(tail) }
                return
            }
            if !prefixes[first].isEmpty { try emit(prefixes[first]) }
            if !buffers[first].isEmpty { try emit(buffers[first]); buffers[first] = "" }
        }
    }
}

struct DocumentOutput {
    let sequences: [BatchOutput]
    let gpuSeconds: Double
    let discardedTokens: Int
    let prefilledInputTokens: Int
    let prefillSeconds: Double
    let decodeSeconds: Double
}

extension Batch {
    func translateOrdered(_ parts: [Segment], tokenizer: Tokenizer, limit: Int,
                          emit: @escaping (String) throws -> Void) throws -> DocumentOutput {
        let plan = try DocumentPlan(parts)
        guard (0...8191).contains(limit), plan.inputs.allSatisfy({ $0.count+limit <= scalar.context && $0.allSatisfy({ (0..<107520).contains($0) }) }) else {
            throw Failure.message("Invalid document tokens or context budget")
        }
        var outputs = [BatchOutput](), gpu = 0.0, discarded = 0, prefilled = 0
        var prefillSeconds = 0.0, decodeSeconds = 0.0
        do {
            for start in stride(from:0,to:plan.inputs.count,by:4) {
                let end = min(start+4,plan.inputs.count)
                let sink = try OrderedOutput(tokenizer:tokenizer,prefixes:Array(plan.prefixes[start..<end]),
                                             tail:end == plan.inputs.count ? plan.tail : "",emit:emit)
                var generated = 0
                let inputs = Array(plan.inputs[start..<end])
                let begin = Date.timeIntervalSinceReferenceDate
                let next = try prefill(inputs)
                let prepared = Date.timeIntervalSinceReferenceDate
                let batch = try decodeBatch(next,limits:[Int](repeating:limit,count:end-start),step:{ try self.step($0) },emit:{ lane,id in
                    generated += 1
                    try sink.append(lane,id)
                },finish:{ try sink.finish($0,$1) },stopOnTruncation:true)
                prefillSeconds += prepared-begin
                decodeSeconds += Date.timeIntervalSinceReferenceDate-prepared
                prefilled += inputs.reduce(0) { $0+$1.count }
                outputs += batch; gpu += gpuSeconds
                discarded += generated-batch.reduce(0) { $0+$1.ids.count }
                if batch.last!.reason != "stop" { break }
            }
            return DocumentOutput(sequences:outputs,gpuSeconds:gpu,discardedTokens:discarded,prefilledInputTokens:prefilled,
                                  prefillSeconds:prefillSeconds,decodeSeconds:decodeSeconds)
        } catch {
            reset()
            throw error
        }
    }
}
