import { fireEvent, screen } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { App } from "../App";
import { assessment, json } from "../test/fixtures";
import { renderPage } from "../test/render";

test("renders a complete, understandable investigation", async () => {
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => Promise.resolve(init?.method === "POST" ? json({ application_id: "APP-1044", outcome: "LEGITIMATE", reviewed_at: "2026-08-20T10:40:00Z" }) : json(assessment)));
  vi.stubGlobal("fetch", fetchMock);
  renderPage(<App />, "/applications/APP-1044");
  expect(await screen.findByRole("heading", { name: "APP-1044" })).toBeInTheDocument();
  expect(screen.getByRole("img", { name: "Fraud risk 91 out of 100" })).toBeInTheDocument();
  expect(screen.getByText("Model probability")).toBeInTheDocument();
  expect(screen.getByText("82%")).toBeInTheDocument();
  expect(screen.getAllByText("Device is associated with multiple applicants").length).toBeGreaterThan(0);
  expect(screen.getByText("fraud-xgb-v1")).toBeInTheDocument();
  expect(screen.getAllByText(/mobile/i).length).toBeGreaterThan(0);
  expect(screen.getByText("HOLD AND INVESTIGATE")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Mark legitimate" }));
  expect(await screen.findByText(/outcome recorded/i)).toBeInTheDocument();
  expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining("/applications/APP-1044/review"), expect.objectContaining({ method: "POST" }));
});
