#!/usr/bin/env python3
"""Attach or verify public Candid metadata in a canister Wasm."""

from __future__ import annotations

import argparse
from pathlib import Path

MAGIC = b"\x00asm\x01\x00\x00\x00"
NAME = b"icp:public candid:service"


def read_u32(data: bytes, start: int) -> tuple[int, int]:
    value = 0
    for index in range(5):
        position = start + index
        if position >= len(data):
            raise ValueError("truncated Wasm length")
        byte = data[position]
        value |= (byte & 0x7F) << (index * 7)
        if not byte & 0x80:
            if value > 0xFFFFFFFF:
                raise ValueError("Wasm length exceeds u32")
            return value, position + 1
    raise ValueError("invalid Wasm length")


def write_u32(value: int) -> bytes:
    output = bytearray()
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def sections(wasm: bytes) -> list[tuple[int, bytes]]:
    if not wasm.startswith(MAGIC):
        raise ValueError("not a Wasm module")
    found = []
    position = len(MAGIC)
    while position < len(wasm):
        section_id = wasm[position]
        length, payload_start = read_u32(wasm, position + 1)
        payload_end = payload_start + length
        if payload_end > len(wasm):
            raise ValueError("truncated Wasm section")
        found.append((section_id, wasm[payload_start:payload_end]))
        position = payload_end
    return found


def metadata_content(wasm: bytes) -> list[bytes]:
    values = []
    for section_id, payload in sections(wasm):
        if section_id != 0:
            continue
        length, start = read_u32(payload, 0)
        end = start + length
        if end > len(payload):
            raise ValueError("truncated Wasm custom section name")
        if payload[start:end] == NAME:
            values.append(payload[end:])
    return values


def attach(wasm: bytes, candid: bytes) -> bytes:
    output = bytearray(MAGIC)
    for section_id, payload in sections(wasm):
        if section_id == 0:
            length, start = read_u32(payload, 0)
            end = start + length
            if end > len(payload):
                raise ValueError("truncated Wasm custom section name")
            if payload[start:end] == NAME:
                continue
        output.append(section_id)
        output.extend(write_u32(len(payload)))
        output.extend(payload)
    payload = write_u32(len(NAME)) + NAME + candid
    output.append(0)
    output.extend(write_u32(len(payload)))
    output.extend(payload)
    return bytes(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("embed", "check"))
    parser.add_argument("wasm", type=Path)
    parser.add_argument("candid", type=Path)
    args = parser.parse_args()
    candid = args.candid.read_bytes()
    if not candid or not candid.rstrip().endswith(b"}"):
        raise ValueError("Candid service is empty or incomplete")
    wasm = args.wasm.read_bytes()
    if args.action == "embed":
        args.wasm.write_bytes(attach(wasm, candid))
        wasm = args.wasm.read_bytes()
    if metadata_content(wasm) != [candid]:
        raise ValueError("Wasm public candid:service metadata differs from the generated Candid")


if __name__ == "__main__":
    main()
