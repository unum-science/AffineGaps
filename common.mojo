"""
Primitives shared by every recurrence in this package.

The alignment and folding families have almost nothing in common beyond these: symbol codes, the
device staging helpers, and the shared-memory budget derived from an occupancy target. Anything
that presumes a rotating band or an affine gap belongs to `alignment.mojo`, and anything that
presumes a base pair belongs to `folding.mojo`.
"""

from std.ffi import c_int, c_size_t, external_call
from std.gpu.host.info import GPUInfo
from std.memory import stack_allocation
from std.sys.info import _accelerator_arch, has_accelerator

from max.gpu.host import DeviceBuffer, DeviceContext

from errors import AffineGapsError, ErrorKind

comptime SCORE_DTYPE = DType.int32
comptime SYMBOL_DTYPE = DType.uint8
comptime SUBSTITUTION_DTYPE = DType.int8
comptime OFFSET_DTYPE = DType.uint64
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

comptime STATIC_SHARED_LIMIT = 48 * 1024
"""Bytes a block may hold without opting into the dynamic carve-out."""

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
    been forbidden from touching.
    """
    comptime WORDS = 16
    var mask = stack_allocation[WORDS, UInt64]()
    for index in range(WORDS):
        mask[index] = 0
    if Int(external_call["sched_getaffinity", c_int](c_int(0), c_size_t(WORDS * 8), mask)) != 0:
        return 1
    var total = 0
    for index in range(WORDS):
        total += Int(mask[index].reduce_bit_count())
    return max(total, 1)


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


def target_shared_per_multiprocessor[blocks_per_multiprocessor: Int]() -> Int:
    """Shared memory per multiprocessor on whatever this is being compiled for.

    A `comptime if` elides the untaken branch where a ternary would instantiate both, which is
    what lets the no-accelerator case avoid naming a device at all: it falls back to the static
    carve-out limit, and nothing sized from it can run on such a host anyway.
    """
    comptime if has_accelerator():
        return GPUInfo.from_name[_accelerator_arch()]().shared_memory_per_multiprocessor
    return blocks_per_multiprocessor * STATIC_SHARED_LIMIT


def shared_per_block[blocks_per_multiprocessor: Int]() -> Int:
    """Static shared memory one block may hold with that many resident per multiprocessor.

    The occupancy target is a parameter because the two recurrence families want opposite
    answers: a banded sweep carries its live diagonals in shared memory and wants a large
    carve-out, while a bifurcating sweep carries only a reduction array and would rather have
    more blocks resident.
    """
    var available = target_shared_per_multiprocessor[blocks_per_multiprocessor]()
    return min(available // blocks_per_multiprocessor, STATIC_SHARED_LIMIT)


def max_dynamic_shared[blocks_per_multiprocessor: Int]() -> Int:
    """Dynamic shared memory one block may opt into, which is all of it but the driver's reserve."""
    return target_shared_per_multiprocessor[blocks_per_multiprocessor]() - SHARED_RESERVED


def uniform_matrix(alphabet_size: Int, match_score: Int, mismatch_score: Int) -> List[Scalar[SUBSTITUTION_DTYPE]]:
    """Diagonal substitution matrix, the `match`/`mismatch` path of `_validate_gotoh_arguments`."""
    var matrix = List[Scalar[SUBSTITUTION_DTYPE]](
        length=alphabet_size * alphabet_size, fill=Scalar[SUBSTITUTION_DTYPE](mismatch_score)
    )
    for index in range(alphabet_size):
        matrix[index * alphabet_size + index] = Scalar[SUBSTITUTION_DTYPE](match_score)
    return matrix^


def translate(text: String, alphabet: String) raises AffineGapsError -> List[Scalar[SYMBOL_DTYPE]]:
    """Maps characters to alphabet indices, raising on anything outside the alphabet."""
    var alphabet_bytes = alphabet.as_bytes()
    var text_bytes = text.as_bytes()
    var codes_by_byte = Array[UInt8, 256](fill=UNKNOWN_SYMBOL)
    for index in range(len(alphabet_bytes)):
        codes_by_byte[Int(alphabet_bytes[index])] = UInt8(index)

    var codes = List[Scalar[SYMBOL_DTYPE]](capacity=len(text_bytes))
    for position in range(len(text_bytes)):
        var code = codes_by_byte[Int(text_bytes[position])]
        if code == UNKNOWN_SYMBOL:
            raise AffineGapsError(ErrorKind.UNKNOWN_SYMBOL, text)
        codes.append(Scalar[SYMBOL_DTYPE](code))
    return codes^


def upload[dtype: DType](ctx: DeviceContext, values: ImmSpan[Scalar[dtype], _]) raises -> DeviceBuffer[dtype]:
    """Stages values onto the device, keeping a one-element floor so an empty batch is not a case."""
    var buffer = ctx.enqueue_create_buffer[dtype](max(len(values), 1))
    if len(values) > 0:
        ctx.enqueue_copy(buffer, values)
    return buffer^


def zeroed[dtype: DType](ctx: DeviceContext, count: Int) raises -> DeviceBuffer[dtype]:
    """A device buffer the caller can read before any kernel has written it."""
    var buffer = ctx.enqueue_create_buffer[dtype](max(count, 1))
    ctx.enqueue_memset(buffer, Scalar[dtype](0))
    return buffer^
