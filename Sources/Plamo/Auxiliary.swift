import Foundation
import Metal
import Darwin

final class Auxiliary {
    private let device: MTLDevice
    private let mapping: Mapping
    private var cached = [String: Ref]()
    private(set) var allocatedBytes = 0
    private let tensors: [String: Any]
    private let dataStart: Int
    private let size: Int

    init(path: String, device: MTLDevice) throws {
        self.device = device
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw Failure.message("Trained adapters missing; run scripts/train_auxiliary.py --collect") }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_size > 8, info.st_size < 268_435_456 else {
            throw Failure.message("Invalid adapter size")
        }
        size = Int(info.st_size)
        let page = Int(getpagesize()), length = (size+page-1)/page*page
        guard let pointer = mmap(nil, length, PROT_READ, MAP_PRIVATE, fd, 0), pointer != MAP_FAILED else {
            throw Failure.message("Cannot map adapters")
        }
        let mapping = Mapping(pointer, length)
        self.mapping = mapping
        let count = pointer.loadUnaligned(as: UInt64.self).littleEndian
        guard count <= 1_048_576, count < size-8, (count+8)%8 == 0 else { throw Failure.message("Invalid adapter header") }
        dataStart = Int(count)+8
        guard let header = try JSONSerialization.jsonObject(with: Data(bytes: pointer+8, count: Int(count))) as? [String: Any],
              let metadata = header["__metadata__"] as? [String: String], metadata["format"] == "plamo-aux-v1",
              metadata["teacher_sha256"] == "7c9f0066d166c075bdbe83ada3d444e08bd7a6ee3f345f47a1f3c442aa6898c5",
              metadata["rank"] == "256", let qs = Int(metadata["qsa_steps"] ?? ""), qs > 0,
              let ms = Int(metadata["mtp_steps"] ?? ""), ms > 0 else {
            throw Failure.message("Incompatible trained adapters")
        }
        tensors = header
    }

    func tensor(_ name: String, _ shape: [Int]) throws -> Ref {
        guard let value = tensors[name] as? [String: Any], value["dtype"] as? String == "BF16",
              value["shape"] as? [Int] == shape, let offsets = value["data_offsets"] as? [Int], offsets.count == 2,
              offsets[0] >= 0, offsets[0]%4 == 0, offsets[1] >= offsets[0],
              offsets[1]-offsets[0] == shape.reduce(2,*), offsets[1] <= size-dataStart else {
            throw Failure.message("Invalid adapter tensor: \(name)")
        }
        if let reference = cached[name] { return reference }
        let page = Int(getpagesize()), start = dataStart+offsets[0], end = dataStart+offsets[1]
        let mappedStart = start/page*page, length = (end+page-1)/page*page-mappedStart
        let mapping = self.mapping
        guard let buffer = device.makeBuffer(bytesNoCopy: mapping.pointer+mappedStart, length: length, options: .storageModeShared,
                                             deallocator: { [mapping] _, _ in withExtendedLifetime(mapping) {} }) else {
            throw Failure.message("Cannot share adapter tensor")
        }
        let reference = Ref(buffer: buffer, offset: start-mappedStart)
        cached[name] = reference
        allocatedBytes += length
        return reference
    }
}

func allocateBuffer(_ device: MTLDevice, _ bytes: Int) throws -> Ref {
    guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
        throw Failure.message("Cannot allocate \(bytes) bytes")
    }
    return Ref(buffer: buffer)
}

struct Indexer {
    let projection: Ref, qNorm: Ref, kNorm: Ref
    let rawKeys: Ref, pooledKeys: Ref
    init(_ device: MTLDevice, _ auxiliary: Auxiliary, layer: Int, context: Int) throws {
        projection = try auxiliary.tensor("qsa.\(layer).proj.weight", [640,1536])
        qNorm = try auxiliary.tensor("qsa.\(layer).q_norm.weight", [128])
        kNorm = try auxiliary.tensor("qsa.\(layer).k_norm.weight", [128])
        // A 16-token batch and the unfinished four-token block fit without overwriting needed keys.
        rawKeys = try allocateBuffer(device, min(context,32)*128*2)
        pooledKeys = try allocateBuffer(device, max(1, context/4)*128*2)
    }
}

struct Sparse {
    let budget: Int
    let projected: Ref, queries: Ref, scores: Ref, indices: Ref, counts: Ref, keys: Ref, values: Ref
    init(_ device: MTLDevice, budget: Int, context: Int, chunk: Int) throws {
        self.budget = budget
        projected = try allocateBuffer(device, chunk*640*4)
        queries = try allocateBuffer(device, chunk*512*4)
        scores = try allocateBuffer(device, chunk*max(1, context/4)*4)
        let slots = context > budget ? budget+3 : 1
        indices = try allocateBuffer(device, chunk*slots*4)
        counts = try allocateBuffer(device, chunk*4)
        keys = try allocateBuffer(device, chunk*slots*128*2)
        values = try allocateBuffer(device, chunk*slots*128*2)
    }
}

struct Predictor {
    let up: Ref, down: Ref
    let input: Ref, activated: Ref, hidden: Ref, stepKeys: Ref
    init(_ device: MTLDevice, _ auxiliary: Auxiliary, context: Int) throws {
        up = try auxiliary.tensor("mtp.heads.0.up.weight", [256,3072])
        down = try auxiliary.tensor("mtp.heads.0.down.weight", [1536,256])
        input = try allocateBuffer(device, 3072*4)
        activated = try allocateBuffer(device, 256*4)
        hidden = try allocateBuffer(device, 1536*4)
        stepKeys = try allocateBuffer(device, Runtime.verifyRows*min(context,2048)*128*2)
    }
}
