# Introduction

FuriosaAI's Tensor Contraction Processor (TCP) is a massively parallel AI accelerator targeting inference workloads.
Unlike high-level frameworks like PyTorch and XLA, which abstract away memory layouts and hardware scheduling, TCP exposes direct programmer control without requiring the byte-level reasoning of low-level kernel APIs.

TCP's Virtual Instruction Set Architecture (Virtual ISA, or vISA) is the programming interface that exposes this control.
It lets programmers reason in tensors while directly managing memory allocation and Tensor Unit scheduling.
This manual primarily helps authors guide an AI agent to write, verify, and optimize vISA kernels.
It also provides the technical material that compiler developers need when generating vISA.
[The `furiosa-opt-std` rustdoc](https://docs.rs/crate/furiosa-opt-std) is the authoritative API source for published releases.
This book explains how to make and review kernel design decisions.
Both audiences assume basic Rust familiarity.
See [the language manual](https://doc.rust-lang.org/book/) if needed.

## Read This Book by Task

[Quick Start](./quick-start.md) introduces the base-template kernel pattern and the learning route through Mapping, Moving, Computing, End-to-End Cases, Scheduling, and Tools.

> [!WARNING]
> **Alpha Test Build: Experimental Software**
>
> This software is an early, experimental, and incomplete build intended strictly for technical evaluation and internal testing.
>
> Before using this software for any production work, critical tasks, or for important data, you must consult with Furiosa engineers.
>
> Your feedback is vital to our development.
> Please provide it.

## License

This documentation and the entire `furiosa-opt` repository are licensed under the [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
