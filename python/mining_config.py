"""Pure-Python MiningConfiguration matching the Rust pearl_mining serialization.

Provides PeriodicPattern, MMAType, and MiningConfiguration with byte-exact
to_bytes() matching the Rust zk-pow crate (52-byte output). No Rust cdylib needed.
"""

from __future__ import annotations

import struct
from enum import IntEnum


class MMAType(IntEnum):
    Int7xInt7ToInt32 = 0


class PeriodicPattern:
    """A periodic pattern of indices (3-dimensional arithmetic progression)."""

    NUM_DIMS: int = 3

    def __init__(self, shape: list[tuple[int, int]]):
        assert len(shape) == self.NUM_DIMS, f"shape must have exactly {self.NUM_DIMS} dims"
        self.shape: list[tuple[int, int]] = shape

    def to_bytes(self) -> bytes:
        data = bytearray(2 * self.NUM_DIMS)
        min_stride = 1
        for i, (stride, length) in enumerate(self.shape):
            factor = stride // min_stride
            data[2 * i] = (factor - 1) & 0xFF
            data[2 * i + 1] = (length - 1) & 0xFF
            min_stride = stride * length
        return bytes(data)

    @classmethod
    def from_bytes(cls, data: bytes) -> PeriodicPattern:
        assert len(data) == 2 * cls.NUM_DIMS
        min_stride = 1
        shape = []
        for i in range(cls.NUM_DIMS):
            factor = data[2 * i] + 1
            length = data[2 * i + 1] + 1
            stride = factor * min_stride
            shape.append((stride, length))
            min_stride = stride * length
        return cls(shape)

    @classmethod
    def from_list(cls, indices: list[int]) -> PeriodicPattern:
        p = list(indices)
        shape_vec: list[tuple[int, int]] = []
        while len(p) > 1:
            found = False
            for period in range(1, len(p)):
                if len(p) % period == 0:
                    s = p[period]
                    if all(p[i] + s == p[i + period] for i in range(len(p) - period)):
                        shape_vec.append((s, len(p) // period))
                        p = p[:period]
                        found = True
                        break
            if not found:
                raise ValueError(f"Pattern is not periodic: {indices}")
        shape_vec.reverse()
        period = shape_vec[-1][0] * shape_vec[-1][1] if shape_vec else 1
        while len(shape_vec) < cls.NUM_DIMS:
            shape_vec.append((period, 1))
        return cls(shape_vec)

    def to_list(self) -> list[int]:
        res = [0]
        for stride, length in self.shape:
            new_res = []
            for i in range(length):
                for r in res:
                    new_res.append(r + i * stride)
            res = new_res
        return res

    def size(self) -> int:
        return self.shape[0][1] * self.shape[1][1] * self.shape[2][1]

    def period(self) -> int:
        return self.shape[-1][0] * self.shape[-1][1]


class MiningConfiguration:
    SERIALIZED_SIZE: int = 52
    RESERVED_SIZE: int = 32
    RESERVED_VALUE: bytes = bytes(RESERVED_SIZE)

    def __init__(
        self,
        common_dim: int,
        rank: int,
        mma_type: MMAType = MMAType.Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern | None = None,
        cols_pattern: PeriodicPattern | None = None,
        reserved: bytes | None = None,
    ):
        self.common_dim: int = common_dim
        self.rank: int = rank
        self.mma_type: MMAType = mma_type
        self.rows_pattern: PeriodicPattern = rows_pattern or PeriodicPattern.from_list([0])
        self.cols_pattern: PeriodicPattern = cols_pattern or PeriodicPattern.from_list([0])
        self.reserved: bytes = reserved if reserved is not None else self.RESERVED_VALUE

    def to_bytes(self) -> bytes:
        buf = bytearray()
        buf.extend(struct.pack("<I", self.common_dim))
        buf.extend(struct.pack("<H", self.rank))
        buf.extend(struct.pack("<H", int(self.mma_type)))
        buf.extend(self.rows_pattern.to_bytes())
        buf.extend(self.cols_pattern.to_bytes())
        buf.extend(self.reserved)
        return bytes(buf)

    @classmethod
    def from_bytes(cls, data: bytes) -> MiningConfiguration:
        assert len(data) == cls.SERIALIZED_SIZE
        common_dim = struct.unpack("<I", data[0:4])[0]
        rank = struct.unpack("<H", data[4:6])[0]
        mma_type = MMAType(struct.unpack("<H", data[6:8])[0])
        rows_pattern = PeriodicPattern.from_bytes(data[8:14])
        cols_pattern = PeriodicPattern.from_bytes(data[14:20])
        reserved = data[20:52]
        return cls(common_dim, rank, mma_type, rows_pattern, cols_pattern, reserved)

    def hash_tile_h(self) -> int:
        return self.rows_pattern.size()

    def hash_tile_w(self) -> int:
        return self.cols_pattern.size()

    def dot_product_length(self) -> int:
        return self.common_dim - self.common_dim % self.rank

    @classmethod
    def from_matrix_dims(
        cls,
        k: int,
        rank: int = 128,
        hash_tile_h: int = 16,
        hash_tile_w: int = 16,
    ) -> MiningConfiguration:
        num_hash_rows = hash_tile_h
        num_hash_cols = hash_tile_w
        rows_pattern = PeriodicPattern.from_list(list(range(num_hash_rows)))
        cols_pattern = PeriodicPattern.from_list(list(range(num_hash_cols)))
        return cls(
            common_dim=k,
            rank=rank,
            mma_type=MMAType.Int7xInt7ToInt32,
            rows_pattern=rows_pattern,
            cols_pattern=cols_pattern,
        )


def patterns_for_tile(tile_h: int, tile_w: int, hash_tile_h: int = 16, hash_tile_w: int = 16):
    num_rows = tile_h // hash_tile_h
    num_cols = tile_w // hash_tile_w
    rows_pattern = PeriodicPattern.from_list(list(range(num_rows)))
    cols_pattern = PeriodicPattern.from_list(list(range(num_cols)))
    return rows_pattern, cols_pattern
