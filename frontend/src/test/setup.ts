import "@testing-library/jest-dom/vitest";
import { afterEach, beforeEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

class ResizeObserverMock { observe() {} unobserve() {} disconnect() {} }
beforeEach(() => vi.stubGlobal("ResizeObserver", ResizeObserverMock));
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });
