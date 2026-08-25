"""
Primitives shared by every recurrence in this package.

The alignment and folding families have almost nothing in common beyond these: symbol codes, the
device staging helpers, and the scoring records every recurrence reads. Anything that presumes a
rotating band or an affine gap belongs to `alignment.mojo`, and anything that presumes a base pair
belongs to `folding.mojo`.
"""

from std.ffi import c_int, c_size_t, external_call
from std.gpu.primitives.warp import WARP_SIZE
from std.memory import stack_allocation
from std.sys.info import CompilationTarget, num_logical_cores, size_of

from max.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext

from errors import AffineGapsError, ErrorKind

comptime ScoreDType = DType.int32
comptime SymbolDType = DType.uint8
comptime SubstitutionDType = DType.int8
comptime OffsetDType = DType.uint64
"""
Indexes the concatenated batch tape rather than one sequence, so it is bounded by the sum of every length in the batch
and not by the longest of them.
"""

comptime GAP_BYTE = Byte(ord("-"))
"""The character a gapped alignment prints where a sequence has nothing."""
comptime FALLBACK_LETTER = Byte(ord("A"))
"""The letter an empty alphabet falls back to, so unit-cost alignment always has one symbol."""

comptime UNKNOWN_SYMBOL = UInt8(255)
"""No alphabet reaches 255 symbols, so it doubles as the "not in this alphabet" marker."""

comptime DEFAULT_PROTEINS_ALPHABET = "ARNDCQEGHILKMFPSTWYVBZX"
"""The twenty-three protein letters BLOSUM62 is tabulated over."""

comptime DEFAULT_RNA_ALPHABET = "ACGU"
"""The four RNA bases, ordered so a base doubles as its own index."""

comptime RNA_ALPHABET_SIZE = DEFAULT_RNA_ALPHABET.byte_length()
"""Bases that alphabet emits, which is the stride of every table indexed by one of them."""

comptime OPEN_BYTE = Byte(ord("("))
"""Opens a base pair in dot-bracket notation."""
comptime CLOSE_BYTE = Byte(ord(")"))
"""Closes a base pair in dot-bracket notation."""
comptime UNPAIRED_BYTE = Byte(ord("."))
"""Marks an unpaired position in dot-bracket notation."""

comptime THREADS_PER_BLOCK = 256
"""Threads in a block launched for a block-wide reduction; the strip and tile sweeps launch one warp."""

comptime WARPS_PER_BLOCK = THREADS_PER_BLOCK // WARP_SIZE
"""Warps such a block holds, which is how many partial results a block-wide reduction combines."""

comptime PositionDType = DType.int32
"""Index into a sequence, wide enough for any input either recurrence can hold."""

comptime MAX_ALPHABET_SIZE = 32
"""
Caps the substitution table staged into shared memory. Thirty-two covers the twenty-three protein letters with room to
spare, and costs one kilobyte per block.
"""

comptime NEGATIVE_INFINITY = Int32.MIN // 4


@fieldwise_init
struct Device(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Which hardware serves a call."""

    var identifier: UInt8
    """Which device this names."""
    comptime CPU = Self(0)
    """The serial reference sweep."""
    comptime GPU = Self(1)
    """The parallel sweep, on one accelerator."""


def hardware_threads() -> Int:
    """Threads this process may actually run on, which an affinity mask or a cgroup quota narrows.

    The online CPU count is the wrong answer on a shared machine: it counts cores this process has
    been forbidden from touching. Only Linux exposes such a mask, and `sched_getaffinity` is a
    glibc symbol, so naming it anywhere else fails at link time rather than at run time.
    """

    @parameter
    if CompilationTarget.is_linux():
        comptime WORDS = 16
        var mask = stack_allocation[WORDS, UInt64]()
        for index in range(WORDS):
            mask[index] = 0
        if Int(external_call["sched_getaffinity", c_int](c_int(0), c_size_t(WORDS * 8), mask)) == 0:
            var total = 0
            for index in range(WORDS):
                total += Int(mask[index].reduce_bit_count())
            return max(total, 1)
    return max(Int(num_logical_cores()), 1)


struct Placement(ImplicitlyCopyable, TrivialRegisterPassable):
    """Where a call runs, and which of the machine's resources it may take.

    Reached through `on_cpu` or `on_gpu` rather than field by field, because an accelerator index on
    a run that never reaches an accelerator is a state nothing downstream can honour.
    """

    var device: Device
    """Which hardware serves the call."""
    var gpu_id: Int
    """Which accelerator, always zero under `Device.CPU`."""
    var threads: Int
    """How many host threads a parallel region may take, always at least one."""

    def __init__(out self, device: Device, gpu_id: Int, threads: Int):
        """Normalizes rather than trusts, so a host run cannot carry an accelerator index."""
        self.device = device
        self.gpu_id = max(gpu_id, 0) if device == Device.GPU else 0
        self.threads = max(threads, 1)

    @staticmethod
    def on_cpu(threads: Int) -> Self:
        """The serial sweep. The width still counts, because the linear-space traceback forks."""
        return Self(Device.CPU, 0, threads)

    @staticmethod
    def on_gpu(gpu_id: Int, threads: Int) -> Self:
        """One accelerator, plus the width of the host region the device path forks back to."""
        return Self(Device.GPU, gpu_id, threads)

    @staticmethod
    def default() -> Self:
        """The host sweep across every thread this process may use."""
        return Self.on_cpu(hardware_threads())


@fieldwise_init
struct GpuSpecs(ImplicitlyCopyable, TrivialRegisterPassable):
    """What one accelerator reports about itself, asked once when a scope opens."""

    var shared_memory_per_multiprocessor: Int
    """Bytes of shared memory one multiprocessor holds, which is what bounds a strip's carry."""
    var reserved_memory_per_block: Int
    """The slice of that a block may not opt into, which the card reports rather than us guessing."""
    var largest_allocation: Int
    """The biggest single buffer this device hands out, which is `maxBufferLength` on Metal."""
    var streaming_multiprocessors: Int
    """How many multiprocessors a grid has to fill."""
    var max_blocks_per_multiprocessor: Int
    """How many blocks one multiprocessor holds at once, which is what a level aims to saturate."""


def gpu_specs_fetch(context: DeviceContext) raises -> GpuSpecs:
    """One cold query of the properties every sweep sizes itself from.

    Each is a live driver call, so they are asked together and once.
    """
    var per_multiprocessor = Int(context.get_attribute(DeviceAttribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR))
    var per_block = Int(context.get_attribute(DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN))
    return GpuSpecs(
        per_multiprocessor,
        per_multiprocessor - per_block,
        Int(context.max_single_alloc_size()),
        Int(context.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)),
        Int(context.get_attribute(DeviceAttribute.MAX_BLOCKS_PER_MULTIPROCESSOR)),
    )


struct DeviceScope(Copyable, Movable):
    """One accelerator and the specs it reported, so no sweep asks the driver twice."""

    var context: DeviceContext
    """Where every sweep is enqueued."""
    var specs: GpuSpecs
    """What that device said about itself, asked when this scope was opened."""

    def __init__(out self, gpu_id: Int) raises:
        """Opens the named accelerator and asks it, once, everything routing will need."""
        self.context = DeviceContext(device_id=gpu_id)
        self.specs = gpu_specs_fetch(self.context)


def uniform_matrix(
    alphabet_size: Int, match_score: Int, mismatch_score: Int
) raises AffineGapsError -> List[Scalar[SubstitutionDType]]:
    """Diagonal substitution matrix, refusing scores the table cannot hold rather than wrapping."""
    comptime lowest = Int(Scalar[SubstitutionDType].MIN)
    comptime highest = Int(Scalar[SubstitutionDType].MAX)
    if match_score < lowest or match_score > highest:
        raise AffineGapsError(ErrorKind.INVALID_SCORING, String("match ", match_score))
    if mismatch_score < lowest or mismatch_score > highest:
        raise AffineGapsError(ErrorKind.INVALID_SCORING, String("mismatch ", mismatch_score))
    var matrix = List[Scalar[SubstitutionDType]](
        length=alphabet_size * alphabet_size, fill=Scalar[SubstitutionDType](mismatch_score)
    )
    for index in range(alphabet_size):
        matrix[index * alphabet_size + index] = Scalar[SubstitutionDType](match_score)
    return matrix^


def translate(text: String, alphabet: String) raises AffineGapsError -> List[Scalar[SymbolDType]]:
    """Maps characters to alphabet indices, raising on anything outside the alphabet."""
    var alphabet_bytes = alphabet.as_bytes()
    var text_bytes = text.as_bytes()
    var codes_by_byte = Array[UInt8, 256](fill=UNKNOWN_SYMBOL)
    for index in range(len(alphabet_bytes)):
        codes_by_byte[Int(alphabet_bytes[index])] = UInt8(index)

    var codes = List[Scalar[SymbolDType]](capacity=len(text_bytes))
    for position in range(len(text_bytes)):
        var code = codes_by_byte[Int(text_bytes[position])]
        if code == UNKNOWN_SYMBOL:
            raise AffineGapsError(ErrorKind.UNKNOWN_SYMBOL, text)
        codes.append(Scalar[SymbolDType](code))
    return codes^


def allocate[dtype: DType](scope: DeviceScope, count: Int) raises -> DeviceBuffer[dtype]:
    """The one place a device buffer is created, and so the one place its size is refused.

    A one-element floor keeps an empty batch from being its own case, and the bound comes off the
    specs the scope already holds.
    """
    var elements = max(count, 1)
    var bytes = elements * size_of[Scalar[dtype]]()
    if bytes > scope.specs.largest_allocation:
        raise AffineGapsError(
            ErrorKind.SEQUENCE_TOO_LONG, String(bytes, " bytes over ", scope.specs.largest_allocation)
        )
    return scope.context.enqueue_create_buffer[dtype](elements)


def upload[dtype: DType](scope: DeviceScope, values: ImmSpan[Scalar[dtype], _]) raises -> DeviceBuffer[dtype]:
    """Stages values onto the device, keeping a one-element floor so an empty batch is not a case."""
    var buffer = allocate[dtype](scope, len(values))
    if len(values) > 0:
        scope.context.enqueue_copy(buffer, values)
    return buffer^


def filled[dtype: DType](scope: DeviceScope, count: Int, value: Scalar[dtype]) raises -> DeviceBuffer[dtype]:
    """A device buffer every element of which is `value` before any kernel has written it."""
    var buffer = allocate[dtype](scope, count)
    scope.context.enqueue_memset(buffer, value)
    return buffer^


def zeroed[dtype: DType](scope: DeviceScope, count: Int) raises -> DeviceBuffer[dtype]:
    """The zero fill, which is what a buffer read before it is written usually wants."""
    return filled[dtype](scope, count, Scalar[dtype](0))
