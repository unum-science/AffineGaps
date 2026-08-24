"""The one error type this package raises, so every fallible entry point declares the same one.

Mojo allows at most one error type per function and never widens a typed `raises` into a plain
one, so a second type here would force every caller to catch and convert. The `detail` names the
sequence, the option or the capacity that was at fault, which is what turns a failure into a
diagnosis.
"""


@fieldwise_init
struct ErrorKind(Equatable, ImplicitlyCopyable, TrivialRegisterPassable, Writable):
    """Why a call into the kernels failed."""

    var code: Int32
    """The negative identifier this kind is reported as."""

    comptime UNKNOWN_SYMBOL = Self(-1)
    """A sequence carried a character the alphabet does not name."""
    comptime ALPHABET_TOO_LARGE = Self(-2)
    """The alphabet exceeds the substitution table staged into shared memory."""
    comptime SEQUENCE_TOO_LONG = Self(-3)
    """A table this input needs is larger than one device allocation may be."""
    comptime SCRATCH_TOO_SMALL = Self(-4)
    """Device scratch was sized for a smaller problem than the one dispatched."""
    comptime LENGTH_MISMATCH = Self(-5)
    """Two inputs that must line up position for position do not."""
    comptime INVALID_SCORING = Self(-6)
    """The gap costs or the substitution scores cannot be served together."""
    comptime INVALID_ARGUMENT = Self(-7)
    """An argument named something this build does not offer, or omitted a value."""
    comptime NOT_ASCII = Self(-8)
    """Unit-cost alignment was handed bytes above the ASCII range."""
    comptime INCONSISTENT_TABLE = Self(-9)
    """A traceback found no case reproducing a stored score, so the fill and the walk disagree."""

    def write_to(self, mut writer: Some[Writer]):
        # Every kind names itself, and a kind added without a line says so rather than borrowing
        # the last one's sentence.
        if self == Self.UNKNOWN_SYMBOL:
            writer.write("a character outside the alphabet")
        elif self == Self.ALPHABET_TOO_LARGE:
            writer.write("the alphabet is larger than the staged table")
        elif self == Self.SEQUENCE_TOO_LONG:
            writer.write("the tables this input needs exceed one device allocation")
        elif self == Self.SCRATCH_TOO_SMALL:
            writer.write("the device scratch is too small for this dispatch")
        elif self == Self.LENGTH_MISMATCH:
            writer.write("two inputs that must line up do not")
        elif self == Self.INVALID_SCORING:
            writer.write("the scoring cannot be served")
        elif self == Self.INVALID_ARGUMENT:
            writer.write("an argument was rejected")
        elif self == Self.NOT_ASCII:
            writer.write("unit-cost alignment handles ASCII only")
        elif self == Self.INCONSISTENT_TABLE:
            writer.write("the table and the traceback disagree")
        else:
            writer.write("an unnamed failure")


@fieldwise_init
struct AffineGapsError(Copyable, ImplicitlyCopyable, Writable):
    """What went wrong, and which sequence, option or capacity it was."""

    var kind: ErrorKind
    """Which category of failure occurred."""
    var detail: String
    """The sequence, option or capacity the failure names."""

    def write_to(self, mut writer: Some[Writer]):
        writer.write("AffineGaps: ", self.kind, " [", self.detail, "]")
