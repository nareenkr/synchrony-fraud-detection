import { screen } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { App } from "../App";
import { json } from "../test/fixtures";
import { renderPage } from "../test/render";

test("renders model identity, threshold and evaluation metrics", async () => {
  vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => Promise.resolve(String(input).includes("learning/status") ? json({ reviewed_applications: 12, confirmed_fraud: 4, legitimate: 7, inconclusive: 1, false_positive_reviews: 2, missed_fraud_reviews: 1, reviewed_alerts: 8, false_positive_review_rate: .25, minimum_feedback_required: 100, retraining_recommended: false, governance_status: "COLLECTING_FEEDBACK" }) : json({ model_version: "fraud-xgb-v1", model_name: "XGBoost", feature_schema_version: "features-v1", classifier_threshold: .42, metrics: { recall: .88, false_positive_rate: .04 }, prototype_only: true }))));
  renderPage(<App />, "/model-info");
  expect(await screen.findByRole("heading", { name: "Model information" })).toBeInTheDocument();
  expect(screen.getByRole("heading", { name: "XGBoost" })).toBeInTheDocument();
  expect(screen.getByText("0.420")).toBeInTheDocument();
  expect(screen.getByText("0.880")).toBeInTheDocument();
  expect(screen.getByText(/prototype decision support/i)).toBeInTheDocument();
  expect(await screen.findByRole("heading", { name: "Governed learning loop" })).toBeInTheDocument();
  expect(screen.getByText("12/100")).toBeInTheDocument();
});
