import Foundation
import Darwin
import Metal
import os

struct Options {
    var model = "models/plamo"
    var target = "ja"
    var maxTokens = 256
    var context: Int?
    var repeats = 1
    var text: String?
    var tokens: String?
    var dump: String?
    var qsa = false
    var qsaBudget = 2048
    var mtp = false
    var repetitionPenalty: Float = 1
    var frequencyPenalty: Float = 0
    var noRepeatNGram = 0
    var loopLimit = 0
    var packed = false
    var stream = false
    var segmentTokens = 0
    var batch4 = false
    var tokenize = false
    var decode: String?

    init(_ arguments: [String]) throws {
        var i = 0
        while i < arguments.count {
            let key = arguments[i]
            if key == "--help" {
                print("""
                plamo [--model DIR] [--to ja|en|LANG] [--max-tokens N] [--context N] [--repeat N] [--packed] [--stream] [--segment-tokens N] [--batch4] [--repetition-penalty F] [--frequency-penalty F] [--no-repeat-ngram N] [--loop-limit N] [--qsa] [--qsa-budget N] [--mtp] [TEXT]
                Without TEXT, reads UTF-8 from stdin. Translation goes to stdout; JSON metrics to stderr.
                --stream writes each decoded token as it is generated, without a trailing newline.
                --stream cannot be combined with --repeat > 1.
                --segment-tokens bounds each independently translated prompt (64...8192, default off).
                Segments share model weights, reset KV, and preserve source whitespace between segments.
                --max-tokens applies per segment; a truncated segment stops the pipeline.
                --batch4 experimentally batches up to four segments; requires --segment-tokens.
                --batch4 supports greedy decoding only and context below 8192.
                --repetition-penalty divides the logits of tokens already in context (default 1 = greedy).
                --frequency-penalty subtracts a multiple of each token's context count (default 0 = greedy).
                --no-repeat-ngram forbids a token that would repeat an n-gram already in context (default 0 = off).
                --loop-limit stops with stop_reason "repeat" when the tail repeats a block this many times (default 0 = off).
                Verification: --tokenize TEXT | --decode IDS | --tokens IDS --dump-logits FILE [--max-tokens 0]
                """)
                exit(0)
            }
            if key == "--qsa" { qsa = true; i += 1; continue }
            if key == "--mtp" { mtp = true; i += 1; continue }
            if key == "--packed" { packed = true; i += 1; continue }
            if key == "--stream" { stream = true; i += 1; continue }
            if key == "--batch4" { batch4 = true; i += 1; continue }
            if key == "--tokenize" { tokenize = true; i += 1; continue }
            if key.hasPrefix("--") {
                guard i+1 < arguments.count else { throw Failure.message("Missing value for \(key)") }
                let value = arguments[i+1]
                switch key {
                case "--qsa-budget":
                    guard let n = Int(value), (4...2048).contains(n), n % 4 == 0 else { throw Failure.message("QSA budget must be a multiple of four in 4...2048") }
                    qsaBudget = n; qsa = true
                case "--segment-tokens":
                    guard let n = Int(value), (64...8192).contains(n) else { throw Failure.message("Segment tokens must be in 64...8192") }
                    segmentTokens = n
                case "--model": model = value
                case "--to": target = value
                case "--max-tokens": guard let n = Int(value), (0...32768).contains(n) else { throw Failure.message("Invalid max-tokens") }; maxTokens = n
                case "--context": guard let n = Int(value), (1...32768).contains(n) else { throw Failure.message("Invalid context") }; context = n
                case "--repeat": guard let n = Int(value), (1...100).contains(n) else { throw Failure.message("Invalid repeat count") }; repeats = n
                case "--repetition-penalty": guard let n = Float(value), n > 0, n.isFinite else { throw Failure.message("Repetition penalty must be a positive finite number") }; repetitionPenalty = n
                case "--frequency-penalty": guard let n = Float(value), n >= 0, n.isFinite else { throw Failure.message("Frequency penalty must be a finite nonnegative number") }; frequencyPenalty = n
                case "--no-repeat-ngram": guard let n = Int(value), (0...64).contains(n) else { throw Failure.message("No-repeat n-gram must be in 0...64") }; noRepeatNGram = n
                case "--loop-limit": guard let n = Int(value), (0...1000).contains(n) else { throw Failure.message("Loop limit must be in 0...1000") }; loopLimit = n
                case "--tokens": tokens = value
                case "--decode": decode = value
                case "--dump-logits": dump = value
                default: throw Failure.message("Unknown option: \(key)")
                }
                i += 2
            } else {
                guard text == nil else { throw Failure.message("Pass source text as a single argument or through stdin") }
                text = key; i += 1
            }
        }
        guard !(tokens != nil && text != nil), !(decode != nil && (tokens != nil || text != nil || tokenize)),
              !(tokenize && tokens != nil), !(stream && (tokenize || decode != nil)),
              !(stream && repeats > 1), !(segmentTokens > 0 && (tokenize || decode != nil || tokens != nil || dump != nil)), !(mtp && (repetitionPenalty != 1 || frequencyPenalty != 0 || noRepeatNGram > 1 || loopLimit > 0)) else {
            throw Failure.message("Conflicting input options")
        }
        guard !batch4 || (segmentTokens > 0 && (context ?? 1) < 8192 && !qsa && !mtp && repetitionPenalty == 1 && frequencyPenalty == 0 && noRepeatNGram == 0 && loopLimit == 0) else {
            throw Failure.message("Batch4 requires segmented greedy decoding without QSA, MTP or repetition controls")
        }
        guard maxTokens > 0 || (tokens != nil && dump != nil) else { throw Failure.message("Zero output requires --tokens and --dump-logits") }
    }
}

func parseIDs(_ text: String) throws -> [Int] {
    let parts = text.split(separator: ",", omittingEmptySubsequences: false)
    let ids = parts.compactMap { Int($0) }
    guard !ids.isEmpty, ids.count == parts.count, ids.allSatisfy({ (0..<107520).contains($0) }) else {
        throw Failure.message("Expected comma-separated token IDs in 0..<107520")
    }
    return ids
}

func json(_ value: Any) throws -> Data {
    var data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys, .fragmentsAllowed])
    data.append(10)
    return data
}

@main struct Main {
    static func main() {
        do { try execute() }
        catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            exit(1)
        }
    }

    static func execute() throws {
        let options = try Options(Array(CommandLine.arguments.dropFirst()))
        let needsMetal = !options.tokenize && options.decode == nil
        let device = needsMetal ? OSAllocatedUnfairLock<MTLDevice?>(initialState: nil) : nil
        let ready = needsMetal ? DispatchGroup() : nil
        if let device, let ready {
            ready.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                let prepared = MTLCreateSystemDefaultDevice()
                device.withLock { $0 = prepared }
                ready.leave()
            }
        }
        let tokenizer = try Tokenizer(path: options.model + "/tokenizer.jsonl")
        if let text = options.decode {
            print(try tokenizer.decode(parseIDs(text)))
            return
        }
        let input: [Int]
        var parts = [Segment]()
        if let tokens = options.tokens {
            input = try parseIDs(tokens)
        } else {
            let text: String
            if let source = options.text { text = source }
            else {
                guard let source = String(data: FileHandle.standardInput.readDataToEndOfFile(), encoding: .utf8) else {
                    throw Failure.message("Input is not UTF-8")
                }
                text = source
            }
            if options.tokenize {
                FileHandle.standardOutput.write(try json(tokenizer.encode(text)))
                return
            }
            if options.segmentTokens > 0 {
                let limit = min(options.segmentTokens,(options.context ?? (options.batch4 ? 8191 : 32768))-options.maxTokens)
                parts = try Segment.split(text,target:options.target,limit:limit,tokenizer:tokenizer)
                input = parts.max(by:{ $0.tokens.count < $1.tokens.count })!.tokens
            } else {
                input = tokenizer.encode(try Tokenizer.prompt(text, target: options.target), bos: true)
            }
        }
        if parts.isEmpty { parts = [Segment(text:"",tokens:input)] }
        let context = options.context ?? min(32768, input.count + options.maxTokens)
        guard input.count + options.maxTokens <= context else { throw Failure.message("Input plus output limit exceeds context") }
        guard !options.batch4 || context < 8192 else { throw Failure.message("Batch4 context must be below 8192") }
        let loadStart = Date.timeIntervalSinceReferenceDate
        ready?.wait()
        let runtime = try autoreleasepool {
            try Runtime(directory: options.model, context: context, packed: options.packed,
                        qsaBudget: options.qsa ? options.qsaBudget : nil, mtp: options.mtp,
                        device: device?.withLock { $0 })
        }
        let plan = options.batch4 ? try DocumentPlan(parts) : nil
        let batch: Batch?
        if let plan, plan.inputs.count > 1 { batch = try autoreleasepool { try Batch(runtime) } }
        else { batch = nil }
        let loadSeconds = Date.timeIntervalSinceReferenceDate - loadStart
        var runs = [[String: Any]]()
        var translated = ""
        let controlled = options.repetitionPenalty != 1 || options.frequencyPenalty != 0 || options.noRepeatNGram > 1
        var logits = [Float]()
        for _ in 0..<options.repeats {
            translated = ""
            if let batch, let plan {
                let output = try batch.translateOrdered(parts,tokenizer:tokenizer,limit:options.maxTokens,emit:{ text in
                    if options.stream { FileHandle.standardOutput.write(Data(text.utf8)) }
                    else { translated += text }
                })
                let segments: [[String:Any]] = output.sequences.enumerated().map { index,sequence in
                    ["input_tokens":plan.inputs[index].count,"output_tokens":sequence.ids.count,
                     "output_ids":sequence.ids,"stop_reason":sequence.reason]
                }
                let decoded = output.sequences.reduce(0) { $0+max(0,$1.ids.count-($1.reason == "length" ? 1 : 0)) }
                runs.append([
                    "input_tokens":segments.reduce(0) { $0+($1["input_tokens"] as! Int) },
                    "prefilled_input_tokens":output.prefilledInputTokens,
                    "output_tokens":output.sequences.reduce(0) { $0+$1.ids.count },
                    "output_ids":output.sequences.flatMap(\.ids),"stop_reason":output.sequences.last!.reason,
                    "prefill_seconds":output.prefillSeconds,"decode_seconds":output.decodeSeconds,
                    "decode_tokens_per_second":Double(decoded)/max(output.decodeSeconds,1e-9),"gpu_seconds":output.gpuSeconds,
                    "segments":segments,"segments_total":plan.inputs.count,
                    "segments_completed":output.sequences.filter { $0.reason == "stop" }.count,
                    "discarded_output_tokens":output.discardedTokens,
                    "mtp_proposed_tokens":0,"mtp_accepted_tokens":0,"mtp_verifications":0,
                    "qsa_queries":0,"qsa_selected_keys":0,"qsa_dense_keys":0
                ])
                continue
            }
            var segmentRuns = [[String:Any]]()
            var adjacent = false
            for part in parts {
                if part.tokens.isEmpty {
                    adjacent = false
                    if options.stream { FileHandle.standardOutput.write(Data(part.text.utf8)) }
                    else { translated += part.text }
                    continue
                }
                if adjacent {
                    if options.stream { FileHandle.standardOutput.write(Data("\n".utf8)) }
                    else { translated += "\n" }
                }
                adjacent = true
                let input = part.tokens
                let stream = options.stream ? Tokenizer.Stream(tokenizer) : nil
                func emit(_ id: Int) throws {
                    guard let stream else { return }
                    let text = try stream.push(id)
                    if !text.isEmpty { FileHandle.standardOutput.write(Data(text.utf8)) }
                }
                runtime.reset()
                var generated = [Int]()
                // Controls see generated tokens only, so the source is never penalized and the
                // coherent prefix stays identical to greedy until output repetition appears.
                func sample(_ greedy: Int) -> Int {
                    guard controlled else { return greedy }
                    runtime.logits(into: &logits)
                    return Sampler.select(&logits, history: generated, penalty: options.repetitionPenalty,
                                          noRepeatNGram: options.noRepeatNGram, frequency: options.frequencyPenalty)
                }
                let start = Date.timeIntervalSinceReferenceDate
                var next = sample(try runtime.prefill(input))
                let prefillEnd = Date.timeIntervalSinceReferenceDate
                if let dump = options.dump {
                    let values = runtime.logitValues()
                    try values.withUnsafeBytes { try Data($0).write(to: URL(fileURLWithPath: dump), options: .atomic) }
                }
                var stopReason: String? = nil
                var proposed = 0, accepted = 0, verifications = 0
                if options.mtp {
                    while generated.count < options.maxTokens {
                        if next == 1 || next == 16 { stopReason = "stop"; break }
                        let remaining = options.maxTokens-generated.count
                        if remaining == 1 { generated.append(next); try emit(next); break }
                        let drafts = try runtime.draft(next)
                        var candidates = [next]
                        for token in drafts.prefix(remaining-1) {
                            if token == 1 || token == 16 { break }
                            candidates.append(token)
                        }
                        proposed += candidates.count-1
                        let startPosition = runtime.position
                        try runtime.forward(candidates, all: true, sequential: true)
                        let verified = runtime.predictions()
                        var count = 1
                        while count < candidates.count && candidates[count] == verified[count-1] { count += 1 }
                        generated.append(contentsOf: candidates.prefix(count))
                        for token in candidates.prefix(count) { try emit(token) }
                        next = verified[count-1]
                        accepted += count-1; verifications += 1
                        try runtime.rewind(startPosition+count, hiddenIndex: count-1)
                    }
                } else {
                    for i in 0..<options.maxTokens {
                        if next == 1 || next == 16 { stopReason = "stop"; break }
                        generated.append(next)
                        try emit(next)
                        if options.loopLimit > 0, Sampler.repetitionPeriod(generated, repeats: options.loopLimit) != nil {
                            stopReason = "repeat"
                            break
                        }
                        if i+1 < options.maxTokens { next = sample(try runtime.forward([next])) }
                    }
                }
                let end = Date.timeIntervalSinceReferenceDate
                let decodeCount = max(0, generated.count - (stopReason == nil ? 1 : 0))
                segmentRuns.append([
                    "input_tokens": input.count, "output_tokens": generated.count,
                    "prefill_seconds": prefillEnd-start, "decode_seconds": end-prefillEnd,
                    "decode_tokens_per_second": Double(decodeCount)/max(end-prefillEnd, 1e-9),
                    "gpu_seconds": runtime.gpuSeconds, "stop_reason": options.maxTokens == 0 ? "prefill" : (stopReason ?? "length"),
                    "output_ids": generated,
                    "mtp_proposed_tokens": proposed, "mtp_accepted_tokens": accepted, "mtp_verifications": verifications,
                    "qsa_queries": runtime.sparseQueries, "qsa_selected_keys": runtime.sparseKeys, "qsa_dense_keys": runtime.denseKeys
                ])
                if let stream { FileHandle.standardOutput.write(Data(stream.finish().utf8)) }
                else { translated += try tokenizer.decode(generated) }
                if stopReason != "stop" { break }
            }
            if options.segmentTokens > 0 {
                var run = segmentRuns[0]
                for key in ["input_tokens","output_tokens","mtp_proposed_tokens","mtp_accepted_tokens","mtp_verifications","qsa_queries","qsa_selected_keys","qsa_dense_keys"] {
                    run[key] = segmentRuns.reduce(0) { $0+($1[key] as! Int) }
                }
                for key in ["prefill_seconds","decode_seconds","gpu_seconds"] {
                    run[key] = segmentRuns.reduce(0.0) { $0+($1[key] as! Double) }
                }
                let decodeTokens = segmentRuns.reduce(0.0) { $0+($1["decode_tokens_per_second"] as! Double)*($1["decode_seconds"] as! Double) }
                run["decode_tokens_per_second"] = decodeTokens/max(run["decode_seconds"] as! Double,1e-9)
                run["output_ids"] = segmentRuns.flatMap { $0["output_ids"] as! [Int] }
                run["stop_reason"] = segmentRuns.last!["stop_reason"]
                run["segments"] = segmentRuns
                run["segments_total"] = parts.filter { !$0.tokens.isEmpty }.count
                run["segments_completed"] = segmentRuns.filter { $0["stop_reason"] as? String == "stop" }.count
                if options.batch4 {
                    run["prefilled_input_tokens"] = run["input_tokens"]
                    run["discarded_output_tokens"] = 0
                }
                runs.append(run)
            } else {
                runs += segmentRuns
            }
        }
        if !options.stream { print(translated) }
        var usage = rusage()
        getrusage(RUSAGE_SELF, &usage)
        var info = rusage_info_v4()
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0)
            }
        }
        guard status == 0 else { throw Failure.message("Cannot read process memory metrics") }
        FileHandle.standardError.write(try json([
            "device": runtime.device.name, "load_seconds": loadSeconds, "packed": options.packed, "qsa": options.qsa, "qsa_budget": options.qsa ? options.qsaBudget : 0, "mtp": options.mtp,
            "repetition_penalty": Double(options.repetitionPenalty), "frequency_penalty": Double(options.frequencyPenalty), "no_repeat_ngram": options.noRepeatNGram,
            "peak_rss_bytes": usage.ru_maxrss,
            "physical_footprint_bytes": info.ri_phys_footprint,
            "peak_physical_footprint_bytes": info.ri_lifetime_max_phys_footprint,
            "resident_bytes": info.ri_resident_size,
            "metal_allocated_bytes": runtime.device.currentAllocatedSize, "peak_metal_allocated_bytes": max(runtime.peakMetalBytes,runtime.device.currentAllocatedSize),
            "gpu_weight_bytes": runtime.weights.gpuBytes, "auxiliary_weight_bytes": runtime.auxiliaryBytes,
            "context": context, "segment_tokens": options.segmentTokens, "batch4": options.batch4, "runs": runs
        ]))
    }
}
