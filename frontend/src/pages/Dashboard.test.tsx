import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import { Dashboard } from "./Dashboard";
import { assessment, json } from "../test/fixtures";
import { renderPage } from "../test/render";

test("renders monitoring KPIs and a clickable live assessment", async () => {
  vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("/applications")) return Promise.resolve(json([assessment]));
    if (url.includes("/analytics")) return Promise.resolve(json({ total_applications: 1, approved_applications: 0, manual_reviews: 0, high_risk_applications: 1, average_risk_score: 91, fraud_high_risk_rate: 1, decision_counts: { APPROVE: 0, MANUAL_REVIEW: 0, HIGH_RISK: 1 } }));
    throw new Error(`Unexpected URL ${url}`);
  }));
  renderPage(<Dashboard />);
  expect(await screen.findByRole("heading", { name: "Fraud monitoring" })).toBeInTheDocument();
  expect(await screen.findByText("APP-1044")).toHaveAttribute("href", "/applications/APP-1044");
  expect(screen.getByText("High risk", { selector: ".decision-badge" })).toBeInTheDocument();
  expect(screen.getByText("91.0")).toBeInTheDocument();
});

test("shows an actionable empty feed", async () => {
  const fetchMock = vi.fn((input: RequestInfo | URL) => Promise.resolve(String(input).includes("analytics") ? json({ total_applications: 0, decision_counts: {} }) : String(input).includes("/demo/") ? json({ running: true }) : json([])));
  vi.stubGlobal("fetch", fetchMock);
  renderPage(<Dashboard />);
  expect(await screen.findByText("Waiting for applications")).toBeInTheDocument();
  const start = screen.getByRole("button", { name: /start demo/i });
  expect(start).toBeEnabled();
  await userEvent.click(start);
  expect(fetchMock).toHaveBeenCalledWith("/api/demo/run", expect.objectContaining({ method: "POST", body: "{}" }));
});

test("configures and starts a seeded random transaction stream", async () => {
  const fetchMock = vi.fn((input: RequestInfo | URL) => Promise.resolve(String(input).includes("analytics") ? json({ total_applications: 0, decision_counts: {} }) : String(input).includes("/demo/") ? json({ running: true }) : json([])));
  vi.stubGlobal("fetch", fetchMock);
  renderPage(<Dashboard />);
  await screen.findByText("Waiting for applications");

  await userEvent.click(screen.getByRole("button", { name: "Random stream" }));
  expect(screen.getByLabelText("Random transaction stream settings")).toBeInTheDocument();
  expect(screen.getByText("Mix 100%")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Start random stream" }));

  expect(fetchMock).toHaveBeenCalledWith(
    "/api/demo/random/run",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({
        count: 100,
        interval_ms: 500,
        seed: 20260820,
        normal_percent: 80,
        suspicious_percent: 15,
        fraud_percent: 5,
      }),
    }),
  );
});
