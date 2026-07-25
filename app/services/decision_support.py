"""Dynamic, XGBoost-backed project decision support.

No recommendation is stored in this module.  Each request explores feasible
values for the controllable model inputs and ranks the resulting predictions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from math import ceil

from app.services.ai_insights import AIInsightsService


# These are feature domains, not action policies.  They mirror the normalized
# feature contract and let the simulator generate feasible alternatives.
CONTROLLABLE_METRICS = (
    ("sprint_velocity", "velocity", 0.0, 1.0, 1),
    ("overdue_rate", "overdue_rate", 0.0, 1.0, -1),
    ("bug_ratio", "bug_ratio", 0.0, 1.0, -1),
    ("communication_sentiment", "overall_sentiment", -1.0, 1.0, 1),
    ("urgency_flag", "urgency_count", 0.0, 1.0, -1),
)
SIMULATION_STEPS = 6


class DecisionSupportService:
    @staticmethod
    def _capacity_risks(capacity_analysis: dict | None) -> dict:
        """Normalise only live capacity evidence into model-ready risk scores."""
        if not capacity_analysis:
            return {"capacity": 0.0, "dependency": 0.0, "burndown": 0.0}
        distribution = capacity_analysis["workload_distribution"]
        members = capacity_analysis["team_capacity"]
        threshold = float(distribution["overload_threshold"])
        capacity = (sum(max(0.0, row["workload_percentage"] - threshold) for row in members)
                    / (len(members) * threshold) if members and threshold else 0.0)
        dependency = (capacity_analysis["dependency_analysis"]["blocked_estimated_hours"] /
                      max(1e-9, capacity_analysis.get("_total_remaining_seconds", 0) / 3600))
        burndown = capacity_analysis["burndown"]["burndown_deviation"]
        # A missing active sprint deliberately uses ``None`` in the dashboard
        # payload.  It means "not measured", never a numeric zero, so the
        # decision engine must not attempt arithmetic with it.
        remaining = capacity_analysis["burndown"]["remaining_story_points"] or 0.0
        return {"capacity": round(min(1.0, max(0.0, capacity)), 4),
                "dependency": round(min(1.0, max(0.0, dependency)), 4),
                "burndown": round(min(1.0, max(0.0, (burndown or 0) / max(1e-9, remaining))), 4)}

    @staticmethod
    def _values(score, values=None):
        return values or {
            "sprint_velocity": float(score.velocity_percent),
            "bug_ratio": float(score.bug_ratio),
            "overdue_rate": float(score.overdue_rate),
            "communication_sentiment": float(score.tone_score),
            "urgency_flag": int(score.urgency_flag),
        }

    @staticmethod
    def _display_name(metric: str) -> str:
        return metric.replace("_", " ").title()

    @classmethod
    def _candidate_values(cls, current: float, lower: float, upper: float, direction: int) -> list[float]:
        """Generate a current-to-best feasible range without policy targets."""
        best = upper if direction > 0 else lower
        if current == best:
            return []
        return [round(current + (best - current) * step / SIMULATION_STEPS, 6)
                for step in range(1, SIMULATION_STEPS + 1)]

    @staticmethod
    def _owner(metric: str) -> str:
        """Map the affected model area to its accountable project role."""
        if metric == "bug_ratio":
            return "Quality lead"
        if metric in {"sprint_velocity", "overdue_rate"}:
            return "Delivery lead"
        if metric == "communication_sentiment":
            return "Project manager"
        return "Project sponsor"

    @classmethod
    def _dependencies(cls, metric: str, values: dict, shap_value: float) -> list[str]:
        """Generate prerequisites from the live affected signal, never a playbook."""
        evidence = f"Validate current {cls._display_name(metric).lower()} measurement"
        model_context = (
            "Review the negative model contribution" if shap_value < 0
            else "Confirm the modelled improvement assumption"
        )
        return [evidence, model_context]

    @staticmethod
    def _effort(change_fraction: float) -> dict:
        """Calculate implementation sizing from the simulated metric delta."""
        score = round(max(0.0, min(1.0, change_fraction)) * 100)
        return {
            "estimated_effort": score,
            "effort": f"{score}/100",
            "estimated_duration_days": max(1, ceil(score / 10)),
            "duration": f"{max(1, ceil(score / 10))} day(s)",
        }

    @classmethod
    def _simulate_metric(cls, project, score, values, metric_spec, email_count, transcript_count, current_score):
        metric, shap_feature, lower, upper, direction = metric_spec
        current = float(values[metric])
        simulations = []
        seen_targets = set()
        for target in cls._candidate_values(current, lower, upper, direction):
            proposed = {**values, metric: int(target) if metric == "urgency_flag" else target}
            # Binary urgency has one meaningful alternative; do not inflate
            # confidence by evaluating the same feature vector repeatedly.
            if proposed[metric] in seen_targets:
                continue
            seen_targets.add(proposed[metric])
            prediction = AIInsightsService.simulate(project, score, proposed, email_count, transcript_count)
            simulations.append({
                "metric": metric, "current": current, "target": proposed[metric],
                "predicted_score": prediction["predicted_score"],
                "gain": round(prediction["predicted_score"] - current_score, 2),
                "prediction_source": prediction["prediction_source"],
                "feature_vector": prediction["feature_vector"],
                "shap_values": prediction["shap_values"],
            })
        if not simulations:
            return None, []
        best = max(simulations, key=lambda item: item["gain"])
        positive_ratio = sum(item["gain"] > 0 for item in simulations) / len(simulations)
        shap_values = best["shap_values"] or {}
        shap_value = float(shap_values.get(shap_feature, 0.0) or 0.0)
        scale = max(upper - lower, 1.0)
        change_fraction = abs(float(best["target"]) - current) / scale
        verb = "Increase" if direction > 0 else "Reduce"
        title = f"{verb} {cls._display_name(metric)} to {best['target']:.2f}"
        action = {
            "action": title, "title": title, "metric": metric,
            "current": current, "target": best["target"], "gain": best["gain"],
            "estimated_improvement": best["gain"], "predicted_score": best["predicted_score"],
            "confidence_score": round(positive_ratio * 100),
            "why_selected": (
                f"{len(simulations)} XGBoost simulations were evaluated; this target produced "
                f"the largest {best['gain']:+.2f} score change. Its current TreeSHAP contribution is {shap_value:+.2f}."
            ),
            "reason": (
                f"Selected from live XGBoost simulations with TreeSHAP contribution {shap_value:+.2f}."
            ),
            "dependencies": cls._dependencies(metric, values, shap_value),
            "owner": cls._owner(metric),
            "affected_metrics": [metric], "affected_developers": [], "affected_issues": [],
            "expected_deadline_improvement_days": 0.0,
            "simulation_count": len(simulations),
            "prediction_source": best["prediction_source"],
            "feature_vector": best["feature_vector"],
        }
        action.update(cls._effort(change_fraction))
        # Existing clients use these compatibility keys.
        action["difficulty"] = action["effort"]
        action["time"] = action["duration"]
        return action, simulations

    @classmethod
    def _delivery_counterfactuals(cls, project, score, values, capacity_analysis, email_count, transcript_count, current_score):
        """Generate capacity/dependency alternatives from the live Jira graph.

        The targets are a consequence of the present workload graph and
        burndown state—not a predefined action catalogue.
        """
        if not capacity_analysis:
            return [], []
        risks = cls._capacity_risks(capacity_analysis)
        members = capacity_analysis["team_capacity"]
        distribution = capacity_analysis["workload_distribution"]
        dependency = capacity_analysis["dependency_analysis"]
        burndown = capacity_analysis["burndown"]
        actions, simulations = [], []

        def evaluate(title, target_velocity, affected_metrics, developers, issues, evidence):
            target_velocity = round(min(1.0, max(values["sprint_velocity"], target_velocity)), 6)
            if target_velocity <= values["sprint_velocity"]:
                return
            proposed = {**values, "sprint_velocity": target_velocity}
            prediction = AIInsightsService.simulate(project, score, proposed, email_count, transcript_count)
            gain = round(prediction["predicted_score"] - current_score, 2)
            simulation = {"metric": "sprint_velocity", "current": values["sprint_velocity"], "target": target_velocity,
                          "predicted_score": prediction["predicted_score"], "gain": gain,
                          "prediction_source": prediction["prediction_source"], "feature_vector": prediction["feature_vector"],
                          "shap_values": prediction["shap_values"], "affected_metrics": affected_metrics,
                          "affected_developers": developers, "affected_issues": issues}
            simulations.append(simulation)
            if gain <= 0:
                return
            effort_fraction = abs(target_velocity - values["sprint_velocity"])
            action = {"action": title, "title": title, "metric": "sprint_velocity", "current": values["sprint_velocity"],
                      "target": target_velocity, "gain": gain, "estimated_improvement": gain,
                      "predicted_score": prediction["predicted_score"], "confidence_score": round((1 - max(risks.values())) * 100),
                      "why_selected": evidence + f" Live XGBoost simulation changed velocity to {target_velocity:.2f} and improved health by {gain:+.2f}.",
                      "reason": evidence, "dependencies": [f"Validate Jira ownership for {', '.join(developers) or 'affected work'}"],
                      "owner": "Delivery lead", "simulation_count": 1, "prediction_source": prediction["prediction_source"],
                      "feature_vector": prediction["feature_vector"], "affected_metrics": affected_metrics,
                      "affected_developers": developers, "affected_issues": issues,
                      "expected_deadline_improvement_days": round((burndown["burndown_deviation"] or 0) * effort_fraction, 2)}
            action.update(cls._effort(effort_fraction))
            action["difficulty"], action["time"] = action["effort"], action["duration"]
            actions.append(action)

        overloaded = [row for row in members if row["developer"] in distribution["overloaded_members"]]
        underutilized = [row for row in members if row["developer"] in distribution["underutilized_members"]]
        if overloaded and underutilized:
            source, destination = overloaded[0], underutilized[0]
            average_hours = sum(row["estimated_remaining_hours"] for row in members) / len(members)
            transferable = min(max(0.0, source["estimated_remaining_hours"] - average_hours),
                               max(0.0, average_hours - destination["estimated_remaining_hours"]))
            relief = transferable / max(1e-9, source["estimated_remaining_hours"])
            evaluate(f"Redistribute {transferable:.2f} remaining hours from {source['developer']} to {destination['developer']}",
                     values["sprint_velocity"] + (1 - values["sprint_velocity"]) * relief,
                     ["developer_capacity", "sprint_velocity"], [source["developer"], destination["developer"]], [],
                     f"Live workload dispersion identifies {source['developer']} above the {distribution['overload_threshold']:.0f}% threshold and {destination['developer']} below the team spread.")
        if dependency["blocked_issues"]:
            recovery = risks["dependency"]
            evaluate(f"Resolve the dependency chain affecting {len(dependency['blocked_issues'])} active Jira issues",
                     values["sprint_velocity"] + (1 - values["sprint_velocity"]) * recovery,
                     ["dependency_risk", "blocked_issues", "sprint_velocity"], dependency["affected_developers"], dependency["blocked_issues"],
                     f"Live Jira links show {dependency['blocked_estimated_hours']:.2f} blocked hours across the dependency graph.")
        if burndown["burndown_deviation"] and burndown["burndown_deviation"] > 0:
            evaluate(f"Recover the {burndown['burndown_deviation']:.2f}-point active-sprint burndown deviation",
                     values["sprint_velocity"] + (1 - values["sprint_velocity"]) * risks["burndown"],
                     ["burndown_deviation", "sprint_velocity"], [], [],
                     f"Actual remaining scope is {burndown['burndown_deviation']:.2f} story points above the live ideal burndown.")
        return actions, simulations

    @classmethod
    def _recovery_plan(cls, project, score, values, recommendations, email_count, transcript_count, current_score):
        """Turn selected actions into sequential, re-predicted execution steps."""
        running_values = dict(values)
        running_score = current_score
        plan = []
        elapsed_days = 0
        for step, action in enumerate(recommendations, start=1):
            running_values[action["metric"]] = action["target"]
            prediction = AIInsightsService.simulate(
                project, score, running_values, email_count, transcript_count
            )
            increase = round(prediction["predicted_score"] - running_score, 2)
            running_score = prediction["predicted_score"]
            elapsed_days += action["estimated_duration_days"]
            prerequisites = list(action["dependencies"])
            if plan:
                prerequisites.append(f"Complete recovery step {step - 1}")
            plan.append({
                "step": step,
                "objective": f"Move {cls._display_name(action['metric']).lower()} from {action['current']} to {action['target']}",
                "owner": action["owner"], "prerequisites": prerequisites,
                "expected_health_increase": increase,
                "cumulative_projected_health": running_score,
                "estimated_completion_timeline": f"Day {elapsed_days}",
                "prediction_source": prediction["prediction_source"],
                # Compatibility fields without duplicating planner prose.
                "action": action["action"], "metric": action["metric"],
                "current": action["current"], "target": action["target"],
                "gain": increase, "duration": action["duration"],
                "dependencies": prerequisites,
            })
        return plan

    @classmethod
    def build(cls, project, score, values=None, email_count=0, transcript_count=0, capacity_analysis=None):
        values = cls._values(score, values)
        baseline = AIInsightsService.simulate(project, score, values, email_count, transcript_count)
        current = baseline["predicted_score"]
        actions, simulations = [], []
        for spec in CONTROLLABLE_METRICS:
            action, metric_simulations = cls._simulate_metric(
                project, score, values, spec, email_count, transcript_count, current
            )
            simulations.extend(metric_simulations)
            if action and action["gain"] > 0:
                actions.append(action)
        delivery_actions, delivery_simulations = cls._delivery_counterfactuals(
            project, score, values, capacity_analysis, email_count, transcript_count, current
        )
        actions.extend(delivery_actions)
        simulations.extend(delivery_simulations)
        # A single dynamic score balances model benefit with the work needed
        # and the recovery of live delivery risk.  The values all originate in
        # the simulation or the current Jira analysis.
        deadline = project.deadline.replace(tzinfo=timezone.utc) if project.deadline.tzinfo is None else project.deadline
        days_remaining = max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds() / 86400)
        for action in actions:
            action["expected_deadline_improvement_days"] = round(
                max(0.0, action["target"] - action["current"]) * days_remaining, 2
            )
        actions.sort(key=lambda item: (
            item["gain"], item["expected_deadline_improvement_days"],
            -item.get("estimated_effort", 0), item.get("confidence_score", 0)
        ), reverse=True)
        actions = actions[:5]
        for priority, action in enumerate(actions, start=1):
            action["priority"] = priority
            action["days_remaining"] = round(days_remaining, 2)
            action["current_deadline"] = deadline.isoformat()
            # Recovery days are the counterfactual velocity delta over the
            # actual remaining calendar.  This stays tied to the simulated
            # model input rather than an arbitrary duration table.

        recovery = cls._recovery_plan(
            project, score, values, actions, email_count, transcript_count, current
        )
        final_values = dict(values)
        for action in actions:
            final_values[action["metric"]] = action["target"]
        horizon = max(1, min(12, int(max(1, (deadline - datetime.now(timezone.utc)).total_seconds() / 604800))))
        forecast = []
        for week in range(horizon + 1):
            fraction = week / horizon
            step_values = {
                key: values[key] + (final_values[key] - values[key]) * fraction
                for key in values if key != "urgency_flag"
            }
            step_values["urgency_flag"] = int(values["urgency_flag"] if fraction < 1 else final_values["urgency_flag"])
            simulated = AIInsightsService.simulate(project, score, step_values, email_count, transcript_count)
            forecast.append({"week": "Current" if week == 0 else f"Week {week}", "score": simulated["predicted_score"],
                             "lower": simulated["predicted_score"], "upper": simulated["predicted_score"]})

        return {
            "current_score": current, "current_health": current,
            "predicted_next_score": forecast[min(1, len(forecast) - 1)]["score"],
            "trend": "Improving" if forecast[-1]["score"] > current else "Stable",
            "trend_reason": "Every point is a fresh XGBoost simulation of the selected actions.",
            "forecast": forecast, "scenarios": actions, "simulation_results": simulations,
            "action_planner": actions, "recommendations": actions, "recovery_plan": recovery,
            "decision_simulator": {"metrics": values, "prediction": current,
                                   "shap_values": baseline["shap_values"], "feature_vector": baseline["feature_vector"]},
            "capacity_risk_scores": cls._capacity_risks(capacity_analysis),
            "risk_level": baseline["rag_status"], "metrics": {
                "tone_score": values["communication_sentiment"], "urgency_flag": values["urgency_flag"],
                "velocity_percent": values["sprint_velocity"], "overdue_rate": values["overdue_rate"],
                "bug_ratio": values["bug_ratio"],
            },
            "risk_reason": "Recommendations are ranked by fresh XGBoost simulations and their TreeSHAP evidence.",
            "primary_causes": [action["metric"] for action in actions],
            "scenario_futures": {"best_case": forecast[-1]["score"], "expected_case": forecast[-1]["score"], "worst_case": current},
            "projected_score_at_deadline": forecast[-1]["score"], "timeline_horizon_weeks": horizon,
        }
