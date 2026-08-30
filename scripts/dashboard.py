#!/usr/bin/env python
"""Build and serve the two-sided demo site.

    uv run python scripts/dashboard.py            # build and serve on :8000
    uv run python scripts/dashboard.py --build    # build only
    uv run python scripts/dashboard.py --port 9000

Two views over one batch run.

**The customer portal.** One test account per failure disposition — six in the
taxonomy, and the builder refuses to ship fewer, because a batch that happens to
miss a disposition would silently produce a demo that never reaches that branch.
A customer picks one, sees what happened in plain language, and
asks for something — retry now, send a link, take it after payday, fix the
mandate, stop messaging me. Every answer comes from the real compliance gate
and the real priced candidate.

**The operations console.** Every request with its verdict and the rule that
bound it; accept, defer and decline rates; what the model actually weighs when
it grades an action; and the batch's money reconciliation and exception list.

Nothing in the page recomputes a decision. Every verdict, probability and
rupee figure is produced here, in Python, by the shipped modules, and embedded
as data — because a second implementation in JavaScript would be a second thing
to keep in agreement, and this project has twice shipped two things that were
supposed to agree and did not.
"""

from __future__ import annotations

import argparse
import datetime as dt
import http.server
import json
import socketserver
import sys
import webbrowser
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rebound.comms import (  # noqa: E402
    CHANNEL_FOR_ACTION,
    Language,
    MerchantProfile,
    MessageBrief,
)
from rebound.compliance import ComplianceGate, Verdict  # noqa: E402
from rebound.desk import CommsDesk, TemplateDrafter  # noqa: E402
from rebound.eval.baselines import default_baselines  # noqa: E402
from rebound.eval.harness import build_eval_batch, evaluate_all  # noqa: E402
from rebound.eval.splits import all_splits  # noqa: E402
from rebound.portal import (  # noqa: E402
    REQUEST_LABELS,
    CustomerRequest,
    answer,
)
from rebound.sequencer import Sequencer, fit_for_serving  # noqa: E402
from rebound.sim.dataset import GenerationConfig, generate_log  # noqa: E402
from rebound.sim.world import World  # noqa: E402
from rebound.taxonomy import Disposition, get_mode  # noqa: E402

OUT = ROOT / "dashboard"

MERCHANT = MerchantProfile(
    name="Vahan Technologies Private Limited",
    short_name="Vahan",
    support_number="18002670001",
    link_host="pay.vahan.in",
    sender_id="VAHANX",
)

#: Test accounts. Names are invented and the episodes behind them are real —
#: one per disposition, so a demo reaches every branch rather than the happy one.
ACCOUNT_NAMES = [
    ("Ananya Rao", "Fitness Pro, monthly"),
    ("Rohit Menon", "Cloud Storage 2TB"),
    ("Priya Nair", "News Daily, annual"),
    ("Imran Sheikh", "Music Premium"),
    ("Kavya Iyer", "Meal Kit, weekly"),
    ("Vikram Bose", "Insurance top-up"),
]


def rupees(paise: float) -> str:
    return f"₹{paise / 100:,.2f}".replace(".00", "")


def build(episodes: int, customers: int, seed: int) -> dict:
    print("Generating a log and fitting the pricer …", flush=True)
    log = generate_log(
        GenerationConfig(
            n_customers=customers,
            start=dt.date(2025, 1, 1),
            end=dt.date(2026, 3, 31),
            seed=20260821,
        )
    )
    pricer = fit_for_serving(all_splits(log)["time"].train, max_iter=120)

    world = World(seed=seed)
    people = world.sample_customers(customers)
    mandates = world.sample_mandates(
        people, dt.date(2024, 6, 1), dt.date(2026, 3, 31)
    )
    world.calibrate(
        people,
        mandates,
        [dt.date(2026, 4, 1) + dt.timedelta(days=i) for i in range(14)],
    )
    batch = build_eval_batch(
        world,
        people,
        mandates,
        dt.date(2026, 4, 1),
        dt.date(2026, 6, 30),
        max_episodes=episodes,
    )
    if not batch:
        raise SystemExit("the simulator produced no failures for this seed")

    accounts = _accounts(world, batch, pricer)
    # A count stated in prose drifts; a count checked against the taxonomy does
    # not. The demo exists to reach every branch, so a batch that surfaced only
    # five dispositions would quietly produce a site that cannot show the sixth.
    missing = {str(d) for d in Disposition} - {a["disposition"] for a in accounts}
    if missing:
        raise SystemExit(
            f"this batch produced no episode for {sorted(missing)}; the demo "
            f"would silently omit that branch.\n"
            f"merchant_fix has one failure code out of 27, so it needs a large "
            f"batch: try --episodes 1400 (the default) or --customers 2000."
        )
    print(f"Running {len(batch):,} episodes through every policy …", flush=True)
    policies = _policies(world, batch, pricer)
    return {
        "generated_at": dt.datetime.now().strftime("%d %b %Y, %H:%M"),
        "merchant": MERCHANT.signature,
        "support": MERCHANT.support_number,
        "episodes": len(batch),
        "accounts": accounts,
        "policies": policies,
        "grading": _grading(pricer),
        "requests": _request_totals(accounts),
    }


def _accounts(world: World, batch, pricer) -> list[dict]:
    """One test account per disposition, with every request pre-answered."""
    gate = ComplianceGate()
    desk = CommsDesk(drafter=TemplateDrafter())
    chosen: dict[Disposition, object] = {}
    for spec in batch:
        episode = world.open_episode(
            f"EP_{len(chosen):04d}",
            spec.mandate,
            spec.customer,
            spec.failure_code,
            spec.failed_at,
            spec.cycles_elapsed,
        )
        view = episode.view()
        chosen.setdefault(view.disposition, view)

    accounts = []
    for index, (disposition, view) in enumerate(sorted(chosen.items(), key=lambda kv: str(kv[0]))):
        if index >= len(ACCOUNT_NAMES):
            break
        name, plan = ACCOUNT_NAMES[index]
        now = view.failed_at + dt.timedelta(hours=20)
        mode = get_mode(view.failure_code)

        answers = []
        for request in CustomerRequest:
            verdict = answer(view, request, now, gate=gate, pricer=pricer)
            answers.append(
                {
                    "request": str(request),
                    "label": REQUEST_LABELS[request],
                    "outcome": str(verdict.outcome),
                    "headline": verdict.headline,
                    "detail": verdict.detail,
                    "action": str(verdict.action) if verdict.action else None,
                    "adjudicated": verdict.adjudicated,
                    "rulings": _rulings(verdict),
                    "grading": _candidate(verdict),
                    "happens_at": (
                        verdict.happens_at.strftime("%d %b %Y, %H:%M")
                        if verdict.happens_at
                        else None
                    ),
                }
            )

        accounts.append(
            {
                "id": f"acct-{index}",
                "name": name,
                "plan": plan,
                "bank": view.bank,
                "rail": str(view.rail),
                "amount": rupees(view.cycle_amount_paise),
                "amount_paise": view.cycle_amount_paise,
                "mandate": view.mandate_id,
                "failed_at": view.failed_at.strftime("%d %b %Y, %H:%M"),
                "failure_code": view.failure_code,
                "failure_label": mode.label,
                "disposition": str(view.disposition),
                "plain": _plain_english(view),
                "messages": _messages(view, gate, desk, now),
                "answers": answers,
            }
        )
    return accounts


def _plain_english(view) -> str:
    """What happened, without a rail code in it."""
    plain = {
        Disposition.RETRY_TIMING: (
            "There was not enough in the account when we tried. Nothing is "
            "wrong with your setup."
        ),
        Disposition.RETRY_TRANSIENT: (
            "Your bank could not be reached when we tried. This one is on "
            "their side, not yours."
        ),
        Disposition.CUSTOMER_ACTION: (
            "Your payment method needs something from you before this can go "
            "through."
        ),
        Disposition.MANDATE_REPAIR: (
            "Your autopay instruction no longer covers this payment and needs "
            "updating."
        ),
        Disposition.MERCHANT_FIX: (
            "This one is our fault — we missed a step we owe you before "
            "collecting."
        ),
        Disposition.TERMINAL: (
            "This autopay instruction has been cancelled, so we cannot collect "
            "on it any more."
        ),
    }[view.disposition]
    # TERMINAL does not mean the mandate is dead. UPI_RETRY_LIMIT_EXCEEDED is
    # terminal for this cycle only, and telling that customer their instruction
    # is cancelled is the error taxonomy.py warns about in its own note.
    if view.disposition is Disposition.TERMINAL and view.mandate_alive:
        return (
            "This payment has been tried the maximum number of times allowed "
            "this cycle. Your autopay is still active — we will collect it "
            "with your next cycle."
        )
    return plain


def _messages(view, gate: ComplianceGate, desk: CommsDesk, now) -> list[dict]:
    """The messages this customer would actually receive, in three languages."""
    from rebound.comms import segments_for
    from rebound.taxonomy import Action, legal_actions

    action = next(
        (
            a
            for a in (Action.NUDGE_SMS, Action.SEND_COLLECT_LINK, Action.NUDGE_WHATSAPP)
            if a in legal_actions(view.failure_code)
        ),
        None,
    )
    if action is None:
        return []
    decision = gate.adjudicate(
        __import__("rebound.compliance", fromlist=["Request"]).Request.from_view(
            view, action, now
        ),
        record=False,
    )
    if decision.approval is None:
        return []

    out = []
    for language in Language:
        brief = MessageBrief.build(
            decision.approval,
            view,
            merchant=MERCHANT,
            language=language,
            retry_on=(now + dt.timedelta(days=2)).date(),
            link=(
                "https://pay.vahan.in/r/7Kd2Qm"
                if action is Action.SEND_COLLECT_LINK
                else None
            ),
        )
        result = desk.compose(brief)
        if result.sent is None:
            continue
        segments, units, encoding = segments_for(result.sent.body)
        out.append(
            {
                "language": str(language),
                "channel": str(CHANNEL_FOR_ACTION[action]),
                "body": result.sent.body,
                "segments": segments,
                "units": units,
                "encoding": encoding,
                "fell_back": result.fell_back,
            }
        )
    return out


def _rulings(verdict) -> list[dict]:
    if verdict.gate is None:
        return []
    return [
        {
            "rule": r.rule_id,
            "basis": str(r.basis),
            "verdict": str(r.verdict),
            "reason": r.reason,
            "binding": verdict.gate.binding is not None
            and r.rule_id == verdict.gate.binding.rule_id,
        }
        for r in verdict.gate.rulings
    ] or [
        {
            "rule": "—",
            "basis": "—",
            "verdict": "allow",
            "reason": "no rule objected",
            "binding": True,
        }
    ]


def _candidate(verdict) -> dict | None:
    c = verdict.candidate
    if c is None:
        return None
    return {
        "p_recover": round(c.p_recover, 4),
        "p_revoke": round(c.p_revoke, 4),
        "recovery_value": round(c.recovery_value_paise / 100, 2),
        "revocation_charge": round(c.revocation_charge_paise / 100, 2),
        "fatigue": round(c.fatigue_externality_paise / 100, 2),
        "cost": round(c.cost_paise / 100, 2),
        "expected_value": round(c.expected_value_paise / 100, 2),
        "worth_unprompted": verdict.worth_doing_unprompted,
    }


def _policies(world: World, batch, pricer) -> dict:
    sequencer = Sequencer(pricer=pricer)
    sequencer.name = "rebound_sequencer"
    results = evaluate_all(
        world, list(default_baselines()) + [sequencer], batch, timeout_seconds=900.0
    )
    failed = {n: r.error for n, r in results.items() if r.error}
    if failed:
        raise SystemExit(f"a policy failed, refusing to build a table: {failed}")

    ladder = results["fixed_ladder"].report.net_rupees_per_1000
    rows = []
    for name, result in results.items():
        r = result.report
        rows.append(
            {
                "policy": name,
                "recovery": round(r.recovery_rate, 4),
                "revocation": round(r.revocation_rate, 4),
                "contacts": round(r.contacts_per_episode, 2),
                "net": round(r.net_rupees_per_1000),
                "gap": round(r.net_rupees_per_1000 - ladder),
                "recovered": round(r.recovered_paise / 100),
                "spent": round(r.spent_paise / 100),
                "destroyed": round(r.destroyed_paise / 100),
            }
        )
    rows.sort(key=lambda row: row["net"], reverse=True)

    ours = results["rebound_sequencer"]
    reasons = Counter(row["reason"].split(":")[0] for row in ours.exceptions)
    audit = [
        {
            "episode": e.episode_id,
            "step": e.step,
            "at": e.at.strftime("%d %b %H:%M"),
            "action": e.action,
            "reason": e.reason,
            "succeeded": e.succeeded,
            "revoked": e.revoked,
            "cost": round(e.cost_paise / 100, 2),
            "recovered": round(e.recovered_paise / 100, 2),
            "detail": e.detail,
        }
        for e in ours.audit[:1200]
    ]
    return {
        "rows": rows,
        "ladder_net": round(ladder),
        "exceptions": [
            {"reason": reason, "count": count, "share": round(count / max(1, len(ours.exceptions)), 4)}
            for reason, count in reasons.most_common()
        ],
        "unrecovered": len(ours.exceptions),
        "audit": audit,
    }


def _grading(pricer) -> dict:
    """What the model actually weighs. Read off the fitted object, not typed in."""
    heads = getattr(pricer, "heads", None)
    features = []
    spec = getattr(getattr(heads, "downstream", None), "spec_", None)
    if spec is not None:
        features = list(spec.columns)
    return {
        "features": features,
        "passive_revocation_rate": round(
            getattr(pricer, "passive_revocation_rate", 0.0), 4
        ),
        "formula": (
            "EV  =  p_recover × (amount + h·D)  −  p_revoke × D(1−h)"
            "  −  fatigue  −  cost"
        ),
        "terms": [
            ["p_recover", "Action head — probability this episode ends recovered."],
            ["p_revoke", "Revocation head — probability this action itself loses the mandate."],
            ["amount + h·D", "What one recovery is worth: the cycle, plus the churn it prevents."],
            ["D(1−h)", "What a revocation costs beyond the churn that was coming anyway."],
            ["fatigue", "The externality this contact imposes on every later contact."],
            ["cost", "Rail fee or channel cost, priced on the rail it presents on."],
        ],
    }


def _request_totals(accounts: list[dict]) -> dict:
    outcomes = Counter()
    by_rule = Counter()
    for account in accounts:
        for a in account["answers"]:
            outcomes[a["outcome"]] += 1
            for r in a["rulings"]:
                if r["binding"] and r["rule"] != "—":
                    by_rule[f"{r['rule']} ({r['verdict']})"] += 1
    return {
        "outcomes": dict(outcomes),
        "binding_rules": by_rule.most_common(),
        "total": sum(outcomes.values()),
    }


# ==========================================================================


def write_site(data: dict) -> Path:
    OUT.mkdir(exist_ok=True)
    (OUT / "data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    page = (Path(__file__).parent / "dashboard_template.html").read_text(
        encoding="utf-8"
    )
    page = page.replace("/*__DATA__*/null", json.dumps(data))
    target = OUT / "index.html"
    target.write_text(page, encoding="utf-8")
    return target


def serve(port: int, open_browser: bool) -> None:
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(OUT), **kw)

        def log_message(self, *a):  # quiet
            pass

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        print(f"\n  Rebound demo site → {url}\n  Ctrl-C to stop.\n")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  stopped.")


def _report_demo_coverage(data: dict) -> None:
    """Say which showcase cases this build actually contains.

    The strongest thing to put on camera is an action the model priced
    *positively* and a regulator refused anyway — permission beating profit,
    in one row. Whether it exists depends on the batch, and finding that out
    while recording is too late. So the build reports it, the same way the
    disposition guard refuses a batch that would silently omit a branch.
    """
    found: dict[str, list[str]] = {
        "regulator overruled a profitable action": [],
        "permitted but not worth doing (done anyway)": [],
        "deferred rather than denied": [],
        "refused before the gate (structural)": [],
    }
    for account in data["accounts"]:
        for answer_ in account["answers"]:
            grading = answer_["grading"]
            binding = next(
                (r for r in answer_["rulings"] if r["binding"] and r["rule"] != "—"),
                None,
            )
            where = f"{account['name']} · {answer_['label']}"
            if (
                answer_["outcome"] == "declined"
                and binding
                and binding["basis"] == "regulatory"
                and grading
                and grading["expected_value"] > 0
            ):
                found["regulator overruled a profitable action"].append(where)
            if grading and not grading["worth_unprompted"] and answer_["outcome"] != "declined":
                found["permitted but not worth doing (done anyway)"].append(where)
            if answer_["outcome"] == "scheduled":
                found["deferred rather than denied"].append(where)
            if answer_["outcome"] == "declined" and not answer_["adjudicated"]:
                found["refused before the gate (structural)"].append(where)

    print("\n  Demo coverage — what this build can show on camera:")
    for label, hits in found.items():
        if hits:
            print(f"    [ok]      {label}")
            print(f"              e.g. {hits[0]}")
        else:
            print(f"    [MISSING] {label}")
    if not found["regulator overruled a profitable action"]:
        print(
            "\n    The strongest case is absent from this batch. Try another "
            "seed:\n      uv run python scripts/dashboard.py --seed 20260904"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1400)
    parser.add_argument("--customers", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--build", action="store_true", help="build without serving")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    data = build(args.episodes, args.customers, args.seed)
    target = write_site(data)
    print(f"\n  Built {target.relative_to(ROOT)} "
          f"({len(data['accounts'])} accounts, {data['episodes']:,} episodes)")
    _report_demo_coverage(data)
    if args.build:
        return 0
    serve(args.port, not args.no_open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
