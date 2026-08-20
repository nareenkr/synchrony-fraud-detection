import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import { json } from "../test/fixtures";
import { renderPage } from "../test/render";
import { NewApplication } from "./NewApplication";

test("submits manually entered loan data through the prediction API", async () => {
  const fetchMock = vi.fn((_input: RequestInfo | URL, _init?: RequestInit) => Promise.resolve(json({ application_id: "APP-MANUAL-001", risk_score: 27, decision: "APPROVE" })));
  vi.stubGlobal("fetch", fetchMock);
  const user = userEvent.setup();
  renderPage(<NewApplication />, "/applications/new");

  await user.type(screen.getByLabelText(/Application ID/), "APP-MANUAL-001");
  await user.type(screen.getByLabelText(/Applicant ID/), "USER-DEMO-001");
  await user.type(screen.getByLabelText(/Requested loan amount/), "5000");
  await user.type(screen.getByLabelText(/Annual income/), "80000");
  await user.selectOptions(screen.getByLabelText(/Lending channel/), "MOBILE");
  await user.click(screen.getByRole("button", { name: "Assess fraud risk" }));

  expect(fetchMock).toHaveBeenCalledOnce();
  const [url, request] = fetchMock.mock.calls[0];
  expect(url).toBe("/api/predict");
  expect(request).toEqual(expect.objectContaining({ method: "POST" }));
  const payload = JSON.parse(String(request?.body));
  expect(payload).toEqual(expect.objectContaining({
    application_id: "APP-MANUAL-001",
    user_id: "USER-DEMO-001",
    channel: "MOBILE",
    requested_loan_amount: 5000,
    income: 80000,
  }));
  expect(payload.event_timestamp).toMatch(/Z$/);
});
