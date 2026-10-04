"""Clearing payout batches (清分付款批次).

The operator already has the immutable clearing ledger (``clearing_batches`` /
``clearing_entries``); the payment system needs to know which amounts have
actually been settled to operators. A payout batch settles the *new* ledger
rows of one or more clearing chains: for every referenced first closing
(``root_clearing_id``, the seq-0 batch) it covers the batches strictly after
the chain's last paid-out seq up to and including the caller-supplied
``expected_seq``.

Safety gates, checked in one transaction, reject the whole payout when:

* a referenced chain does not exist or is not a first closing (404 / 422);
* ``expected_seq`` is not the chain's current latest batch seq -- a stale view
  (newer clearing batches exist) or a future seq the chain has not reached
  (409);
* any ledger row in the covered range was already taken by another payout
  (409) -- each clearing batch of a chain is paid out at most once;
* the caller's per-operator expected net amounts do not match the covered
  rows exactly (operator set and signed cents, 422).

On success the immutable payout snapshot, per-chain coverage and per-operator
detail are persisted together (storage-level triggers forbid UPDATE/DELETE).
The same ``payout_id`` replays idempotently for the identical canonical
payload; a different payload returns 409.

Later clearing adjustment batches only append new ``clearing_entries`` rows;
they never rewrite rows a payout covered. ``GET /clearings/payouts/{id}``
therefore reports, per operator, the frozen payout, the live chain position
and the signed difference (待补付 top-up / 待追回 clawback) without touching
the snapshot.
"""
from __future__ import annotations

import hashlib

from .audit import canonical_json
from .settlement import iso_str


class PayoutError(Exception):
    """Business-rule rejection while planning a payout. ``status_code`` follows
    the endpoint conventions: 404 unknown chain, 409 stale/conflicting seq or
    already-paid ledger rows, 422 malformed references/amounts."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def canonical_payout_payload(body) -> dict:
    """Canonical request: chains sorted by root id, operator expectations
    sorted by operator id, so semantically identical submissions hash
    identically regardless of array order (duplicate ids are 422 upstream)."""
    return {
        "payout_id": body.payout_id,
        "external_payment_ref": body.external_payment_ref,
        "chains": sorted(
            (
                {
                    "root_clearing_id": c.root_clearing_id,
                    "expected_seq": c.expected_seq,
                }
                for c in body.chains
            ),
            key=lambda c: c["root_clearing_id"],
        ),
        "operator_net_amounts": sorted(
            (
                {
                    "operator_id": o.operator_id,
                    "expected_net_amount_cents": o.expected_net_amount_cents,
                }
                for o in body.operator_net_amounts
            ),
            key=lambda o: o["operator_id"],
        ),
    }


def payload_hash(payload: dict) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def expected_manifest(body) -> dict[str, int]:
    """Caller's expected per-operator signed net amounts, keyed by operator."""
    return {
        o.operator_id: o.expected_net_amount_cents for o in body.operator_net_amounts
    }


def chain_last_paid_seq(conn, root_clearing_id: str) -> int | None:
    """Highest clearing batch seq already covered by a payout on this chain,
    or None when the chain has never been paid out."""
    row = conn.execute(
        "SELECT MAX(expected_seq) AS last_seq FROM payout_chains"
        " WHERE root_clearing_id = ?",
        (root_clearing_id,),
    ).fetchone()
    if row is None or row["last_seq"] is None:
        return None
    return row["last_seq"]


def covered_entries(
    conn, root_clearing_id: str, after_seq: int, through_seq: int
) -> list:
    """Immutable clearing ledger rows of one chain with clearing batch seq in
    the half-open range ``(after_seq, through_seq]``, ordered for a stable
    global entry numbering."""
    return conn.execute(
        "SELECT b.clearing_id, b.seq AS clearing_seq, b.kind, b.net_amount_cents,"
        " e.trip_id, e.passenger_id, e.from_station, e.to_station, e.ended_at,"
        " e.operator_id, e.amount_cents"
        " FROM clearing_entries e"
        " JOIN clearing_batches b ON b.clearing_id = e.clearing_id"
        " WHERE b.root_clearing_id = ? AND b.seq > ? AND b.seq <= ?"
        " ORDER BY b.seq, e.seq",
        (root_clearing_id, after_seq, through_seq),
    ).fetchall()


def build_payout_plan(conn, body) -> dict:
    """Plan the whole payout under the caller's transaction. Purely reads
    clearing/payout tables and assembles the snapshot; raises PayoutError on
    the first gate that fails. Raises before any write, so a rejection leaves
    no payout, coverage or operator rows behind."""
    chains_out: list[dict] = []
    entries_out: list[dict] = []
    operator_totals: dict[str, int] = {}

    # Chains are processed in canonical order; the explicit overlap query
    # below is the actual guarantee, but checking last_paid first gives the
    # clearest error for the ordinary "already paid" replay race.
    for chain in sorted(body.chains, key=lambda c: c.root_clearing_id):
        root = conn.execute(
            "SELECT * FROM clearing_batches WHERE clearing_id = ?",
            (chain.root_clearing_id,),
        ).fetchone()
        if root is None:
            raise PayoutError(
                404,
                (
                    f"root_clearing_id '{chain.root_clearing_id}' not found; run"
                    " POST /clearings for the first closing first"
                ),
            )
        if root["kind"] != "initial" or root["seq"] != 0:
            raise PayoutError(
                422,
                (
                    f"root_clearing_id '{chain.root_clearing_id}' is an"
                    " adjustment batch; payouts must reference the first"
                    " closing (seq 0)"
                ),
            )

        latest_seq = conn.execute(
            "SELECT MAX(seq) AS latest FROM clearing_batches"
            " WHERE root_clearing_id = ?",
            (chain.root_clearing_id,),
        ).fetchone()["latest"]

        if chain.expected_seq != latest_seq:
            if chain.expected_seq < latest_seq:
                raise PayoutError(
                    409,
                    (
                        f"expected_seq {chain.expected_seq} for clearing chain"
                        f" '{chain.root_clearing_id}' is stale: the chain's"
                        f" current latest batch seq is {latest_seq}"
                    ),
                )
            raise PayoutError(
                409,
                (
                    f"expected_seq {chain.expected_seq} for clearing chain"
                    f" '{chain.root_clearing_id}' does not exist: the chain's"
                    f" current latest batch seq is {latest_seq}"
                ),
            )

        last_paid = chain_last_paid_seq(conn, chain.root_clearing_id)
        start_seq = -1 if last_paid is None else last_paid
        if chain.expected_seq <= start_seq:
            raise PayoutError(
                409,
                (
                    f"clearing chain '{chain.root_clearing_id}' is already paid"
                    f" out through seq {start_seq}; expected_seq"
                    f" {chain.expected_seq} adds no new ledger rows"
                ),
            )

        # Defence in depth: intersect every earlier payout's own covered
        # interval (pc.last_paid_seq, pc.expected_seq] with this payout's new
        # range (start_seq, expected_seq]. The MAX(expected_seq) watermark
        # already rules overlap out under normal use; this check also catches
        # rows whose stored coverage somehow disagrees with the watermark.
        overlap = conn.execute(
            "SELECT pc.payout_id, b.seq FROM payout_chains pc"
            " JOIN clearing_batches b ON b.root_clearing_id = pc.root_clearing_id"
            " WHERE pc.root_clearing_id = ? AND b.seq > ? AND b.seq <= ?"
            " AND b.seq > pc.last_paid_seq AND b.seq <= pc.expected_seq"
            " ORDER BY b.seq LIMIT 1",
            (chain.root_clearing_id, start_seq, chain.expected_seq),
        ).fetchone()
        if overlap is not None:
            raise PayoutError(
                409,
                (
                    f"clearing batch seq {overlap['seq']} of chain"
                    f" '{chain.root_clearing_id}' is already included in payout"
                    f" '{overlap['payout_id']}'; a clearing batch is paid out at"
                    " most once"
                ),
            )

        rows = covered_entries(
            conn, chain.root_clearing_id, start_seq, chain.expected_seq
        )
        if not rows:
            raise PayoutError(
                409,
                (
                    f"no new clearing ledger rows to pay out on chain"
                    f" '{chain.root_clearing_id}' after seq {start_seq}"
                ),
            )

        chain_net = 0
        for r in rows:
            chain_net += r["amount_cents"]
            operator_totals[r["operator_id"]] = (
                operator_totals.get(r["operator_id"], 0) + r["amount_cents"]
            )
            entries_out.append(
                {
                    "root_clearing_id": chain.root_clearing_id,
                    "clearing_id": r["clearing_id"],
                    "clearing_seq": r["clearing_seq"],
                    "kind": r["kind"],
                    "trip_id": r["trip_id"],
                    "passenger_id": r["passenger_id"],
                    "from_station": r["from_station"],
                    "to_station": r["to_station"],
                    "ended_at": r["ended_at"],
                    "operator_id": r["operator_id"],
                    "amount_cents": r["amount_cents"],
                }
            )
        covered = sorted({r["clearing_seq"] for r in rows})
        chains_out.append(
            {
                "root_clearing_id": chain.root_clearing_id,
                "expected_seq": chain.expected_seq,
                "last_paid_seq": start_seq,
                "covered_seqs": covered,
                "entry_count": len(rows),
                "net_amount_cents": chain_net,
            }
        )

    # The manifest must cover exactly the ledger-derived net amounts: same
    # operator set and same signed cents for every operator.
    actual = dict(operator_totals)
    expected = expected_manifest(body)
    missing = sorted(set(actual) - set(expected))
    extra = sorted(set(expected) - set(actual))
    mismatched = sorted(
        op for op in actual.keys() & expected.keys() if expected[op] != actual[op]
    )
    problems: list[str] = []
    if missing:
        problems.append(
            f"operator net amounts missing for {missing}"
            f" (ledger computes { {o: actual[o] for o in missing} })"
        )
    if extra:
        problems.append(
            f"unexpected operator net amounts for {extra}"
            " (no covered ledger rows for those operators)"
        )
    if mismatched:
        problems.append(
            "operator net amount mismatch for "
            + ", ".join(
                f"{o}: expected {expected[o]}, ledger {actual[o]}" for o in mismatched
            )
        )
    if problems:
        raise PayoutError(422, "; ".join(problems))

    chains_out.sort(key=lambda c: c["root_clearing_id"])
    entries_out.sort(
        key=lambda e: (
            e["root_clearing_id"],
            e["clearing_seq"],
            e["trip_id"],
            e["operator_id"],
        )
    )
    return {
        "chains": chains_out,
        "entries": entries_out,
        "operator_totals": dict(sorted(operator_totals.items())),
        "net_amount_cents": sum(operator_totals.values()),
    }


def payout_response(
    *,
    payout_id: str,
    external_payment_ref: str,
    plan: dict,
    status: str,
    created_at: str,
) -> dict:
    """Assemble the API body (also frozen verbatim into payouts.result)."""
    return {
        "payout_id": payout_id,
        "status": status,
        "external_payment_ref": external_payment_ref,
        "chain_count": len(plan["chains"]),
        "entry_count": len(plan["entries"]),
        "net_amount_cents": plan["net_amount_cents"],
        "chains": [
            {
                "seq": seq,
                "root_clearing_id": c["root_clearing_id"],
                "expected_seq": c["expected_seq"],
                "last_paid_seq": c["last_paid_seq"],
                "covered_seqs": c["covered_seqs"],
                "entry_count": c["entry_count"],
                "net_amount_cents": c["net_amount_cents"],
            }
            for seq, c in enumerate(plan["chains"])
        ],
        "operator_totals": [
            {"operator_id": op, "amount_cents": amount}
            for op, amount in plan["operator_totals"].items()
        ],
        "entries": [
            {
                "seq": seq,
                "root_clearing_id": e["root_clearing_id"],
                "clearing_id": e["clearing_id"],
                "clearing_seq": e["clearing_seq"],
                "kind": e["kind"],
                "trip_id": e["trip_id"],
                "passenger_id": e["passenger_id"],
                "from_station": e["from_station"],
                "to_station": e["to_station"],
                "ended_at": e["ended_at"],
                "operator_id": e["operator_id"],
                "amount_cents": e["amount_cents"],
            }
            for seq, e in enumerate(plan["entries"])
        ],
        "created_at": created_at,
    }


def load_payout(conn, payout_id: str) -> dict | None:
    return conn.execute(
        "SELECT * FROM payouts WHERE payout_id = ?", (payout_id,)
    ).fetchone()


def _chain_live_totals(conn, root_clearing_id: str) -> dict[str, int]:
    """Live per-operator position of one chain across *every* batch the chain
    currently has (the immutable initial split plus all adjustment deltas)."""
    rows = conn.execute(
        "SELECT e.operator_id, SUM(e.amount_cents) AS amount"
        " FROM clearing_entries e"
        " JOIN clearing_batches b ON b.clearing_id = e.clearing_id"
        " WHERE b.root_clearing_id = ?"
        " GROUP BY e.operator_id",
        (root_clearing_id,),
    ).fetchall()
    return {r["operator_id"]: r["amount"] for r in rows}


def _all_paid_chain_amounts(conn, root_clearing_id: str) -> dict[str, int]:
    """Per-operator amount *every* payout has covered on one chain (all payout
    snapshots, regardless of when they ran). A clearing entry row is copied
    into at most one payout, so this is the chain's total settled position."""
    rows = conn.execute(
        "SELECT operator_id, SUM(amount_cents) AS amount"
        " FROM payout_entries WHERE root_clearing_id = ?"
        " GROUP BY operator_id",
        (root_clearing_id,),
    ).fetchall()
    return {r["operator_id"]: r["amount"] for r in rows}


def payout_status(conn, row) -> dict:
    """Build the live status view for a stored payout. The frozen snapshot
    (``result``) is returned unchanged; the per-chain and per-operator
    top-up/clawback balances are recomputed from the current clearing ledger
    every time, so a clearing adjustment appended after the payout shows up
    here without ever rewriting a paid row.

    The balance view spans every chain this payout covers: for each operator,
    ``current`` is the live chain position (initial split plus all adjustment
    deltas booked so far) and ``paid`` is the amount settled across *all*
    payouts on those chains (a follow-up payout that covers the newer batches
    closes the gap this view reported). Their difference is the outstanding
    amount: positive 待补付, negative 待追回."""
    payout_id = row["payout_id"]
    chain_rows = conn.execute(
        "SELECT * FROM payout_chains WHERE payout_id = ? ORDER BY seq",
        (payout_id,),
    ).fetchall()
    frozen = conn.execute(
        "SELECT operator_id, SUM(amount_cents) AS amount FROM payout_entries"
        " WHERE payout_id = ? GROUP BY operator_id",
        (payout_id,),
    ).fetchall()
    frozen_by_operator = {r["operator_id"]: r["amount"] for r in frozen}

    chains: list[dict] = []
    # Live amounts across the covered chains and everything already paid out
    # on them (including later payouts extending the same chains).
    live_total: dict[str, int] = {}
    paid_total: dict[str, int] = {}
    for c in chain_rows:
        root = c["root_clearing_id"]
        latest_seq = conn.execute(
            "SELECT MAX(seq) AS latest FROM clearing_batches"
            " WHERE root_clearing_id = ?",
            (root,),
        ).fetchone()["latest"]
        last_paid_now = chain_last_paid_seq(conn, root)
        paid_through = -1 if last_paid_now is None else last_paid_now

        live_by_operator = _chain_live_totals(conn, root)
        paid_by_operator = _all_paid_chain_amounts(conn, root)
        for op, amount in live_by_operator.items():
            live_total[op] = live_total.get(op, 0) + amount
        for op, amount in paid_by_operator.items():
            paid_total[op] = paid_total.get(op, 0) + amount

        current_amount = sum(live_by_operator.values())
        paid_amount = sum(paid_by_operator.values())
        chains.append(
            {
                "root_clearing_id": root,
                "paid_through_seq": paid_through,
                "latest_seq": latest_seq,
                "current_amount_cents": current_amount,
                "paid_amount_cents": paid_amount,
                "difference_cents": current_amount - paid_amount,
            }
        )

    operators = sorted(set(frozen_by_operator) | set(live_total) | set(paid_total))
    balances: list[dict] = []
    for op in operators:
        current = live_total.get(op, 0)
        paid = paid_total.get(op, 0)
        difference = current - paid
        if difference > 0:
            direction = "top_up_due"
        elif difference < 0:
            direction = "clawback_due"
        else:
            direction = "none"
        balances.append(
            {
                "operator_id": op,
                "frozen_amount_cents": frozen_by_operator.get(op, 0),
                "paid_amount_cents": paid,
                "current_amount_cents": current,
                "difference_cents": difference,
                "direction": direction,
            }
        )

    return {
        "payout_id": payout_id,
        "status": "created",
        "external_payment_ref": row["external_payment_ref"],
        "chain_count": row["chain_count"],
        "entry_count": row["entry_count"],
        "net_amount_cents": row["net_amount_cents"],
        "chains": chains,
        "operator_balances": balances,
        "created_at": row["created_at"],
    }
