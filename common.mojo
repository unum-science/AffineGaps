"""
Primitives shared by every recurrence in this package.

The alignment and folding families have almost nothing in common beyond these: symbol codes, the
device staging helpers, and the scoring records every recurrence reads. Anything that presumes a
rotating band or an affine gap belongs to `alignment.mojo`, and anything that presumes a base pair
belongs to `folding.mojo`.
"""

from std.ffi import c_int, c_size_t, external_call
from std.memory import stack_allocation
from std.sys.info import CompilationTarget, num_logical_cores, size_of

from max.gpu.host import DeviceBuffer, DeviceContext

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

comptime OPEN_BYTE = Byte(ord("("))
"""Opens a base pair in dot-bracket notation."""
comptime CLOSE_BYTE = Byte(ord(")"))
"""Closes a base pair in dot-bracket notation."""
comptime UNPAIRED_BYTE = Byte(ord("."))
"""Marks an unpaired position in dot-bracket notation."""

comptime THREADS_PER_BLOCK = 256
"""Threads in every block this package launches."""

comptime MAX_ALPHABET_SIZE = 32
"""
Caps the substitution table staged into shared memory. Thirty-two covers the twenty-three protein letters with room to
spare, and costs one kilobyte per block.
"""

comptime SHARED_RESERVED = 1024
"""
A block may opt into all of a multiprocessor's shared memory but the kilobyte the driver keeps. Measured on this
target: 227 kibibytes of dynamic shared memory launches, 228 does not.
"""

comptime NEGATIVE_INFINITY = Int32.MIN // 4


@fieldwise_init
struct Executor(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Where a sweep runs."""

    var identifier: UInt8
    """Which executor this names."""
    comptime HOST = Self(0)
    """The serial reference sweep, on the CPU."""
    comptime DEVICE = Self(1)
    """The parallel sweep, on the GPU."""


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
    """Where a sweep runs, and which of the machine's resources it may take.

    Reached through `host` or `device` rather than field by field, because an accelerator index on a
    run that never reaches an accelerator is a state nothing downstream can honour.
    """

    var executor: Executor
    """Which of the two sweeps serves the call."""
    var gpu_id: Int
    """Which accelerator, always zero under `Executor.HOST`."""
    var threads: Int
    """How many host threads a parallel region may take, always at least one."""

    def __init__(out self, executor: Executor, gpu_id: Int, threads: Int):
        """Normalizes rather than trusts, so a host run cannot carry an accelerator index."""
        self.executor = executor
        self.gpu_id = max(gpu_id, 0) if executor == Executor.DEVICE else 0
        self.threads = max(threads, 1)

    @staticmethod
    def host(threads: Int) -> Self:
        """The serial sweep. The width still counts, because the linear-space traceback forks."""
        return Self(Executor.HOST, 0, threads)

    @staticmethod
    def device(gpu_id: Int, threads: Int) -> Self:
        """One accelerator, plus the width of the host region the device path forks back to."""
        return Self(Executor.DEVICE, gpu_id, threads)

    @staticmethod
    def default() -> Self:
        """The host sweep across every thread this process may use."""
        return Self.host(hardware_threads())


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


def allocate[dtype: DType](ctx: DeviceContext, count: Int) raises -> DeviceBuffer[dtype]:
    """The one place a device buffer is created, and so the one place its size is refused.

    A one-element floor keeps an empty batch from being its own case. The bound is the largest
    contiguous allocation this device reports, which on Metal is a buffer limit well below the
    card's memory rather than the memory itself.
    """
    var elements = max(count, 1)
    var bytes = elements * size_of[Scalar[dtype]]()
    var largest = Int(ctx.max_single_alloc_size())
    if bytes > largest:
        raise AffineGapsError(ErrorKind.SEQUENCE_TOO_LONG, String(bytes, " bytes over ", largest))
    return ctx.enqueue_create_buffer[dtype](elements)


def upload[dtype: DType](ctx: DeviceContext, values: ImmSpan[Scalar[dtype], _]) raises -> DeviceBuffer[dtype]:
    """Stages values onto the device, keeping a one-element floor so an empty batch is not a case."""
    var buffer = allocate[dtype](ctx, len(values))
    if len(values) > 0:
        ctx.enqueue_copy(buffer, values)
    return buffer^


def filled[dtype: DType](ctx: DeviceContext, count: Int, value: Scalar[dtype]) raises -> DeviceBuffer[dtype]:
    """A device buffer every element of which is `value` before any kernel has written it."""
    var buffer = allocate[dtype](ctx, count)
    ctx.enqueue_memset(buffer, value)
    return buffer^


def zeroed[dtype: DType](ctx: DeviceContext, count: Int) raises -> DeviceBuffer[dtype]:
    """The zero fill, which is what a buffer read before it is written usually wants."""
    return filled[dtype](ctx, count, Scalar[dtype](0))
