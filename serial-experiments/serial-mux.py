import pty
from threading import Thread, Lock
from queue import Queue
import os
import struct
from hashlib import blake2s
from typing_extensions import Buffer
import serial
import select

HASH_SIZE = 32
PACK_SIZE = f"!x{HASH_SIZE}s"
test = blake2s(b"3289hh29gj4g3").digest()


class MuxPacket:
    data: bytes

    def __init__(self, data: bytes, target: int):
        self.data = data
        self.target = target

    def __bytes__(self):
        length = len(self.data)
        data = struct.pack(f"!BI{length}s", self.target, length, self.data)

        return (
            struct.pack(PACK_SIZE, blake2s(data, digest_size=HASH_SIZE).digest()) + data
        )

    @staticmethod
    def FromBytes(input: Buffer):
        (hash,) = struct.unpack_from(PACK_SIZE, input, 0)

        if hash != blake2s(input[HASH_SIZE + 1 :], digest_size=HASH_SIZE).digest():
            raise RuntimeError("Checksum failed!")

        (target, length) = struct.unpack_from("!BI", input, HASH_SIZE + 1)

        (data,) = struct.unpack_from(
            f"!{length}s", input, HASH_SIZE + 1 + struct.calcsize("!BI")
        )

        return MuxPacket(data, target)


class MuxRouter:
    ids: list[int]

    out: serial.Serial

    def __init__(self, output: serial.Serial, ports: int):
        self.out = output

        self.queue = Queue(1)

        for index in range(ports):
            (internal, external) = pty.openpty()

            print(f"ID {index}: {os.ttyname(external)}")

            self.ids[index] = internal

        Thread()

    def run(self):
        reads, writes, excepts = select.select(self.ids, self.ids, [], 0)

        for readable in reads:
            self.read_virtual(readable)


    def read_serial(self):
        while True:
            data = self.out.read_until()

            try:
                packet = MuxPacket.FromBytes(data)

                output = self.out[packet.target]

                os.write(output, packet.data)
            except RuntimeError:
                continue

    def do_write_serial(self):
        while True:
            self.out.write(self.queue.get())

    def read_virtual(self, id: int):
        while True:
            os.read(self.ids[id])

    def write_serial(self, data, timeout):
        self.queue.put()

