/**
 * @file Shared interfaces and code for the low-level backend API.
 *
 * Think of each backend as a _connector_ to a specific hardware or software
 * implementation of the array API.
 *
 * Backends do not share any of the built-in operational semantics of the
 * library. This is a private API. You must allocate and free buffers manually,
 * and dispatch happens on the level of each shader. Buffers are untyped.
 */

export async function getBackend(backendName: string): Promise<Backend | null> {
  if (backendName === "cpu") {
    const { CPUBackend } = await import("./backend/cpu");
    return new CPUBackend();
  } else if (backendName === "webgpu") {
    const { WebGPUBackend } = await import("./backend/webgpu");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    return new WebGPUBackend(device);
  } else {
    throw new Error(`Backend not found: ${backendName}`);
  }
}

/** Unique identifier for an allocated, on-device buffer. */
export type Slot = number;

/** A device backend. */
export interface Backend {
  /** Allocate a new slot with reference count 1. */
  malloc(size: number, initialData?: ArrayBuffer): Slot;

  /** Increment the reference count of the slot. */
  incRef(slot: Slot): void;

  /**
   * Decrement the reference count of the slot. If the reference count reaches
   * zero, it is freed. This should throw if the slot was already freed.
   */
  decRef(slot: Slot): void;

  /** Read a range of bytes from a buffer. */
  read(slot: Slot, start?: number, count?: number): Promise<ArrayBuffer>;

  /** Read a range of bytes from a buffer, blocking variant. */
  readSync(slot: Slot, start?: number, count?: number): ArrayBuffer;

  /** Run a backend operation. */
  executeOp(op: BackendOp, inputs: Slot[], outputs: Slot[]): Promise<void>;

  /** Run a backend operation, blocking variant. */
  executeOpSync(op: BackendOp, inputs: Slot[], outputs: Slot[]): void;
}

export enum BackendOp { // TODO: This is temporary
  Add,
  Mul,
}

export class SlotError extends Error {
  constructor(slot: Slot) {
    super(`Used a buffer that is invalid or already freed: ${slot}`);
  }
}
