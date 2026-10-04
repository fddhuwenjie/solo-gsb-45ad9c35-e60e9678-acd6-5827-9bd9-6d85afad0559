"""Clearing payout batches (清分付款批次).

Operators already have the clearing ledger; the payment system needs to know
which amounts have actually been settled. A *payout batch*
(``POST /clearings/payouts``) submits a unique ``payout_id``, one or more
first-closing ``clearing_id`` chains, each chain's ``expected_seq``, a
per-operator expected net manifest and an external payment reference.

Inside a single transaction the service reads, for every referenced chain,
the ledger rows booked after the chain's last paid seq up to ``expected_seq``
(``expected_seq`` must be the chain's current latest batch) and sums them per
operator. The submission is rejected as a whole when a chain does not exist,
the seq is stale, the pending rows are already covered by another payout, or
the manifest does not equal the computed non-zero per-operator nets exactly.

On success the immutable payout snapshot (batch row, per-chain covered range,
per-operator nets) is written together with the audit record. Later clearing
adjustments only append new ledger rows and never rewrite paid records; the
payout status query recomputes the live per-operator balance (current chain
position minus everything paid) and reports it as ``top_up_due`` (待补付) or
``clawback_due`` (待追回).
"""
from __future__ import annotations

import hashlib

from .audit import canonical_json
from .models import (
    ClearingPayoutBalance,
    ClearingPayoutChainOut,
    ClearingPayoutOperatorOut,
    ClearingPayoutOut,
)


def canonical_payout_payload(body) -> dict:
    """Canonical payout request: chains sorted by clearing id and each chain's
    operator manifest sorted by operator id, so semantically identical
    submissions hash identically regardless of array order."""
    return {
        "payout_id": body.payout_id,
        "external_ref": body.external_ref,
        "chains": sorted(
            (
                {
                    "clearing_id": chain.clearing_id,
                    "expected_seq": chain.expected_seq,
                    "operators": sorted(
                        (
                            {
                                "operator_id": op.operator_id,
                                "amount_cents": op.amount_cents,
                            }
                            for op in chain.operators
                        ),
                        key=lambda o: o["operator_id"],
                    ),
                }
                for chain in body.chains
            ),
            key=lambda c: c["clearing_id"],
        ),
    }


def payload_hash(payload: dict) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def new_entry_totals(
    conn, root_clearing_id: str, from_seq: int, to_seq: int
) -> tuple[dict[str, int], int]:
    """Per-operator signed net of the chain's ledger rows in the half-open
    batch range ``(from_seq, to_seq]``, plus the number of rows covered.
    Operators whose rows net to zero are kept here; the caller decides
    whether to drop them."""
    rows = conn.execute(
        "SELECT e.operator_id, SUM(e.amount_cents) AS amount, COUNT(*) AS n"
        " FROM clearing_entries e"
        " JOIN clearing_batches b ON b.clearing_id = e.clearing_id"
        " WHERE b.root_clearing_id = ? AND b.seq > ? AND b.seq <= ?"
        " GROUP BY e.operator_id",
        (root_clearing_id, from_seq, to_seq),
    ).fetchall()
    totals = {r["operator_id"]: r["amount"] for r in rows}
    return totals, sum(r["n"] for r in rows)


def paid_operator_totals(conn, root_clearing_id: str) -> dict[str, int]:
    """Cumulative net already paid per operator by every payout batch that
    covers this chain."""
    rows = conn.execute(
        "SELECT operator_id, SUM(amount_cents) AS amount"
        " FROM clearing_payout_operators"
        " WHERE root_clearing_id = ? GROUP BY operator_id",
        (root_clearing_id,),
    ).fetchall()
    return {r["operator_id"]: r["amount"] for r in rows}


def live_chain_totals(conn, root_clearing_id: str) -> dict[str, int]:
    """Current chain-wide booked position per operator (every batch of the
    chain, including adjustments appended after the payouts)."""
    rows = conn.execute(
        "SELECT e.operator_id, SUM(e.amount_cents) AS amount"
        " FROM clearing_entries e"
        " JOIN clearing_batches b ON b.clearing_id = e.clearing_id"
        " WHERE b.root_clearing_id = ? GROUP BY e.operator_id",
        (root_clearing_id,),
    ).fetchall()
    return {r["operator_id"]: r["amount"] for r in rows}


def chain_balances(conn, root_clearing_id: str) -> list[ClearingPayoutBalance]:
    """Live payment balance per operator: the chain's current booked position
    minus everything already paid. Positive -> ``top_up_due`` (待补付),
    negative -> ``clawback_due`` (待追回), zero -> ``none``."""
    paid = paid_operator_totals(conn, root_clearing_id)
    current = live_chain_totals(conn, root_clearing_id)
    balances = []
    for operator_id in sorted(set(paid) | set(current)):
        p = paid.get(operator_id, 0)
        c = current.get(operator_id, 0)
        difference = c - p
        balances.append(
            ClearingPayoutBalance(
                operator_id=operator_id,
                paid_cents=p,
                current_cents=c,
                difference_cents=difference,
                direction=(
                    "top_up_due" if difference > 0
                    else "clawback_due" if difference < 0
                    else "none"
                ),
            )
        )
    return balances


def payout_response(conn, payout_row, status: str) -> ClearingPayoutOut:
    """Build the API view for a stored payout. The frozen snapshot (covered
    ranges and per-operator nets) comes from the append-only rows; the
    per-chain balances are recomputed live on every read, so adjustments
    booked after the payout surface as top-up/clawback differences without
    ever touching the frozen rows. ``status`` is the request replay marker
    ("created" or "duplicate")."""
    chains = conn.execute(
        "SELECT seq, root_clearing_id, from_seq, to_seq, entry_count,"
        " amount_cents FROM clearing_payout_chains"
        " WHERE payout_id = ? ORDER BY seq",
        (payout_row["payout_id"],),
    ).fetchall()
    operators = conn.execute(
        "SELECT chain_seq, operator_id, amount_cents"
        " FROM clearing_payout_operators"
        " WHERE payout_id = ? ORDER BY chain_seq, seq",
        (payout_row["payout_id"],),
    ).fetchall()
    ops_by_chain: dict[int, list[ClearingPayoutOperatorOut]] = {}
    for o in operators:
        ops_by_chain.setdefault(o["chain_seq"], []).append(
            ClearingPayoutOperatorOut(
                operator_id=o["operator_id"], amount_cents=o["amount_cents"]
            )
        )
    return ClearingPayoutOut(
        payout_id=payout_row["payout_id"],
        status=status,
        external_ref=payout_row["external_ref"],
        chain_count=payout_row["chain_count"],
        total_amount_cents=payout_row["total_amount_cents"],
        chains=[
            ClearingPayoutChainOut(
                clearing_id=c["root_clearing_id"],
                from_seq=c["from_seq"],
                to_seq=c["to_seq"],
                entry_count=c["entry_count"],
                amount_cents=c["amount_cents"],
                operators=ops_by_chain.get(c["seq"], []),
                balances=chain_balances(conn, c["root_clearing_id"]),
            )
            for c in chains
        ],
        created_at=payout_row["created_at"],
    )
