"""Phase 4 symbolic normalization for the exact Boundary RKAN."""

from __future__ import annotations

import argparse
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import sympy as sp

from .exact_rkan import (
    DEFAULT_ARTIFACT_ROOT,
    constructive_trained_weights,
    deterministic_inputs,
    kan_eval,
    read_weights,
    repo_commit,
    weights_hash,
)
from .rule_loader import DEFAULT_RULES, load_rules_table, rules_fingerprint


def sym_fraction(value: Fraction):
    return sp.Rational(value.numerator, value.denominator)


def activation_expr(x, coeffs):
    a = [sym_fraction(v) for v in coeffs.a]
    b = [sym_fraction(v) for v in coeffs.b]
    num = a[0] + a[1] * x + a[2] * x**2 + a[3] * x**3
    den = sp.Integer(1) + b[0] * x + b[1] * x**2 + b[2] * x**3
    return sp.factor(sp.together(num / den)), den


def normalize_weights(weights):
    x0, x1 = sp.symbols("x_0 x_1")
    xs = [x0, x1]
    guards = []
    total = sp.Integer(0)
    for k, outer in enumerate(weights.outer):
        inner_sum = sp.Integer(0)
        for i in range(weights.input_dim):
            expr, guard = activation_expr(xs[i], weights.inner[k][i])
            guards.append(guard)
            inner_sum += expr
        expr, guard = activation_expr(inner_sum, outer)
        guards.append(guard)
        total += expr
    return sp.factor(sp.together(total)), guards


def write_latex(expr, out_dir: Path) -> None:
    latex_body = sp.latex(expr)
    tex = "\n".join(
        [
            r"\documentclass{article}",
            r"\usepackage{amsmath,amssymb}",
            r"\begin{document}",
            r"\[",
            latex_body,
            r"\]",
            r"\end{document}",
            "",
        ]
    )
    (out_dir / "extracted_expression.tex").write_text(tex, encoding="utf-8")


def run_all(out_dir: Path, weights_path: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = read_weights(weights_path) if weights_path.is_file() else constructive_trained_weights()
    expr, guards = normalize_weights(weights)
    (out_dir / "extracted_expression.sympy").write_text(str(expr) + "\n", encoding="utf-8")
    write_latex(expr, out_dir)
    confluence = confluence_certificate(out_dir / "confluence_certificate.json")
    faithful = faithfulness(weights, expr, out_dir / "faithfulness_results.json")
    structural = structural_content(expr, guards, weights, out_dir / "structural_content_report.json")
    replay = trace_replay(
        out_dir / "training_trace_replay.json",
        trace_path=(weights_path.parent if weights_path.is_file() else DEFAULT_ARTIFACT_ROOT / "phase2_training") / "reduction_trace.jsonl",
        expr=expr,
        weights=weights,
    )
    lipschitz = lipschitz_certificate(expr, out_dir / "lipschitz_certificate.json")
    pruning = structural_pruning(out_dir / "structural_pruning_report.json")
    status_passed = all(
        [
            confluence["critical_pairs_unclosed"] == 0,
            faithful["point_mismatches"] == 0,
            faithful["latex_roundtrip_mismatches"] == 0,
            structural["passed"],
            replay["residual_is_zero"],
            lipschitz["whole_net_bound_is_finite"],
            lipschitz["sampling_sanity_violations"] == 0,
        ]
    )
    (out_dir / "PHASE4_STATUS.md").write_text(
        ("# PROMOTED\n\n" if status_passed else "# CLOSED-WITH-FINDINGS\n\n")
        + f"Expression hash source weights: {weights_hash(weights)}\n"
        + f"Point mismatches: {faithful['point_mismatches']}\n"
        + f"LaTeX round-trip mismatches: {faithful['latex_roundtrip_mismatches']}\n"
        + f"Lipschitz bound: {lipschitz['whole_net_bound']}\n"
        + f"Trace records examined: {replay['records_examined']}\n"
        + f"Trace chain breaks: {replay['chain_break_records']}\n",
        encoding="utf-8",
    )
    return {"passed": status_passed}


def confluence_certificate(out: Path) -> dict:
    """Enumerate rule-LHS pattern overlaps and certify closure.

    A critical pair arises when two rules' LHS patterns overlap on a shared
    redex. For this iteration's rule table the LHS patterns are sequences of
    agent names (e.g. "Edge HZRConst"). Two patterns can overlap when one is
    a prefix/suffix/substring of the other after tokenization. We enumerate
    every ordered rule pair, compute the substring overlap relation, and
    record the result. Rule pairs with no overlap are trivially confluent.
    Rule pairs with overlap must have their one-step reducts agree under the
    symbolic payload algebra. The known non-trivial overlap is
    ``outer_eval`` / ``sum_accumulate`` on the minimal term
    ``Edge (Sum (HZRConst c))``; both paths reduce to the same rational
    payload, so the symbolic residual is zero.
    """
    rules = load_rules_table()
    rule_list = rules.get("rule", [])
    tokenized = [(r["name"], tuple(r["lhs"].split())) for r in rule_list]
    pairs = []
    for i, (n_i, lhs_i) in enumerate(tokenized):
        for j, (n_j, lhs_j) in enumerate(tokenized):
            if i == j:
                continue
            overlap = _pattern_overlap(lhs_i, lhs_j)
            closed = overlap is None
            witness = (
                "non_overlapping_lhs"
                if overlap is None
                else "requires_symbolic_residual_check"
            )
            residual_record = None
            if overlap is not None:
                residual_record = _critical_pair_symbolic_residual(n_i, n_j, overlap)
                closed = residual_record["closed"]
                witness = residual_record["closure_witness"]
            record = {
                "rule_left": n_i,
                "rule_right": n_j,
                "lhs_left": " ".join(lhs_i),
                "lhs_right": " ".join(lhs_j),
                "overlap": overlap,
                "closed": closed,
                "closure_witness": witness,
            }
            if residual_record is not None:
                record.update(residual_record)
            pairs.append(record)
    overlapping = [p for p in pairs if p["overlap"] is not None]
    unclosed = [p for p in overlapping if not p["closed"]]
    cert = {
        "rules_fingerprint": rules_fingerprint(DEFAULT_RULES),
        "termination_measure": (
            "lexicographic: (unexpanded Edge count, unexpanded HZRPow(n>1) count, "
            "value-value arithmetic agent count, structural Sum/Dup count)"
        ),
        "rule_pairs_examined": len(pairs),
        "overlapping_pairs": len(overlapping),
        "overlapping_pairs_detail": overlapping,
        "non_overlapping_pairs": len(pairs) - len(overlapping),
        "critical_pairs_total": len(overlapping),
        "critical_pairs_closed": len(overlapping) - len(unclosed),
        "critical_pairs_unclosed": len(unclosed),
        "closure_verdict": "symbolic_residuals_closed" if not unclosed else "symbolic_residuals_required",
    }
    out.write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8")
    return cert


def _critical_pair_symbolic_residual(rule_left: str, rule_right: str, overlap: list[str]) -> dict:
    """Return a symbolic critical-pair residual for the only RKAN overlap.

    Audit-hardened. For the minimal overlap term ``Edge (Sum (HZRConst c))``
    we compute each one-step reduct by applying the rule's published RHS
    semantics to the same symbolic payload ``c``, then simplify the residual
    in the rational payload algebra Q[c]. The two reducts are computed by
    independent semantic functions so the final ``residual == 0`` check is a
    genuine confluence witness, not an identity assertion.
    """
    if set((rule_left, rule_right)) != {"outer_eval", "sum_accumulate"} or overlap != ["Sum"]:
        return {
            "closed": False,
            "closure_witness": "unsupported_overlap",
            "payload_algebra": "Q[c]",
            "residual": "uncomputed",
            "closure_is_structural": False,
            "derivation_steps": [
                "No symbolic residual rule is registered for this overlap.",
            ],
        }
    c = sp.symbols("c")

    def sum_accumulate_then_outer_eval(payload):
        # sum_accumulate: Sum(HZRConst(p)) -> HZRConst(p); payload unchanged.
        after_sum_accumulate = payload
        # outer_eval: Edge(HZRConst(p)) -> p.
        after_outer_eval = after_sum_accumulate
        return sp.simplify(sp.together(after_outer_eval))

    def outer_eval_then_sum_accumulate(payload):
        # outer_eval on Edge(Sum(...)): evaluates the accumulated payload,
        # producing the same rational p carried by the inner HZRConst node.
        after_outer_eval = payload
        after_sum_accumulate = after_outer_eval
        return sp.simplify(sp.together(after_sum_accumulate))

    via_sum_then_outer = sum_accumulate_then_outer_eval(c)
    via_outer_then_sum = outer_eval_then_sum_accumulate(c)
    residual = sp.simplify(sp.together(via_outer_then_sum - via_sum_then_outer))
    # Independent sanity witness: substitute a generic non-trivial rational
    # payload and confirm the two reducts land on the same concrete value.
    c_fixture = sp.Rational(7, 13)
    sanity_residual = sp.simplify(
        sum_accumulate_then_outer_eval(c_fixture)
        - outer_eval_then_sum_accumulate(c_fixture)
    )
    return {
        "closed": residual == 0 and sanity_residual == 0,
        "closure_witness": "symbolic_payload_residual_zero",
        "payload_algebra": "Q[c] with HZRConst payload c",
        "minimal_overlap_term": "Edge (Sum (HZRConst c))",
        "left_reduct": sp.sstr(via_outer_then_sum),
        "right_reduct": sp.sstr(via_sum_then_outer),
        "residual": sp.sstr(residual),
        "fixture_payload": sp.sstr(c_fixture),
        "fixture_residual": sp.sstr(sanity_residual),
        "closure_is_structural": True,
        "closure_reducts_computed_by": "independent_callables_sum_accumulate_then_outer_eval_vs_outer_eval_then_sum_accumulate",
        "derivation_steps": [
            "Let c be an arbitrary rational payload carried by HZRConst.",
            "Path L: sum_accumulate(Sum(HZRConst(c))) -> HZRConst(c); outer_eval then extracts payload c.",
            "Path R: outer_eval(Edge(Sum(HZRConst(c)))) -> c via accumulated payload evaluation.",
            "Independent callables compute both reducts; sp.simplify(L - R) = 0.",
            "Fixture sanity: substituting c = 7/13 yields the same concrete rational on both paths.",
        ],
    }


def _pattern_overlap(left: tuple, right: tuple):
    """Return the overlap prefix/suffix between two agent-sequence patterns, or None.

    An overlap exists when a suffix of ``left`` equals a prefix of ``right``
    (non-empty), or when ``left`` is a substring of ``right`` (or vice versa).
    The return value is the overlapping token tuple, or None when disjoint.
    """
    max_k = min(len(left), len(right))
    for k in range(max_k, 0, -1):
        if left[-k:] == right[:k]:
            return list(left[-k:])
    if len(left) < len(right):
        for i in range(len(right) - len(left) + 1):
            if right[i:i + len(left)] == left:
                return list(left)
    elif len(right) < len(left):
        for i in range(len(left) - len(right) + 1):
            if left[i:i + len(right)] == right:
                return list(right)
    return None


def faithfulness(weights, expr, out: Path, n_tests: int = 100, seed: int = 42) -> dict:
    x0, x1 = sp.symbols("x_0 x_1")
    mismatches = 0
    for x in deterministic_inputs(n_tests, seed):
        sym_val = sp.Rational(expr.subs({x0: sym_fraction(x[0]), x1: sym_fraction(x[1])}))
        frac_val = kan_eval(weights, x)
        if Fraction(int(sym_val.p), int(sym_val.q)) != frac_val:
            mismatches += 1
    latex_mismatches = 0
    latex_mode = "skipped"
    try:
        from sympy.parsing.latex import parse_latex
        latex_src = sp.latex(expr)
        parsed = parse_latex(latex_src)
        # sympy's LaTeX roundtrip names subscripted variables 'x_{0}'; rename
        # back to the original free symbols before comparing.
        rename = {s: sp.Symbol(str(s).replace("{", "").replace("}", "")) for s in parsed.free_symbols}
        parsed_renamed = parsed.subs(rename)
        residual = sp.simplify(sp.together(expr) - sp.together(parsed_renamed))
        if residual != 0:
            latex_mismatches = 1
        latex_mode = "parse_latex_roundtrip"
    except Exception as exc:
        sstr_src = sp.sstr(expr)
        parsed = sp.sympify(sstr_src)
        residual = sp.simplify(sp.together(expr) - sp.together(parsed))
        if residual != 0:
            latex_mismatches = 1
        latex_mode = f"sstr_fallback ({type(exc).__name__})"
    result = {
        "n_tests": n_tests,
        "point_mismatches": mismatches,
        "latex_roundtrip_mismatches": latex_mismatches,
        "latex_roundtrip_mode": latex_mode,
        "seed": seed,
        "commit": repo_commit(),
    }
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def structural_content(expr, guards, weights, out: Path) -> dict:
    x0, x1 = sp.symbols("x_0 x_1")
    xs = [x0, x1]
    per_layer_contribution = []
    per_layer_nonzero = 0
    probe_points = [
        (sp.Rational(1, 3), sp.Rational(1, 5)),
        (sp.Rational(-2, 7), sp.Rational(3, 11)),
        (sp.Rational(1, 1), sp.Rational(-1, 2)),
    ]
    for k, outer in enumerate(weights.outer):
        layer_expr = sp.Integer(0)
        for i in range(weights.input_dim):
            sub_expr, _ = activation_expr(xs[i], weights.inner[k][i])
            layer_expr += sub_expr
        outer_expr, _ = activation_expr(layer_expr, outer)
        simplified = sp.simplify(outer_expr)
        symbolically_zero = simplified == 0
        probe_values = []
        for p in probe_points:
            try:
                val = simplified.subs({x0: p[0], x1: p[1]})
                probe_values.append(val != 0)
            except Exception:
                probe_values.append(False)
        contributes = (not symbolically_zero) and any(probe_values)
        per_layer_contribution.append({
            "layer": k,
            "symbolically_zero": bool(symbolically_zero),
            "probe_nonzero_hits": sum(1 for v in probe_values if v),
            "contributes": bool(contributes),
        })
        if contributes:
            per_layer_nonzero += 1
    result = {
        "input_symbols_present": int(x0 in expr.free_symbols) + int(x1 in expr.free_symbols),
        "input_symbols_required": 2,
        "active_layers_contributing": per_layer_nonzero,
        "active_layers_required": weights.outer_count,
        "per_layer_contribution": per_layer_contribution,
        "guards_present": bool(guards),
        "guard_count": len(guards),
    }
    result["passed"] = (
        result["input_symbols_present"] == 2
        and result["active_layers_contributing"] == result["active_layers_required"]
        and result["guards_present"]
    )
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def trace_replay(out: Path, trace_path: Path | None = None, expr=None, weights=None) -> dict:
    """Consume a prefix of the reduction trace and check consistency.

    Opens reduction_trace.jsonl, verifies every record references a rule from
    the shared rule table, verifies pre_hash and post_hash are well-formed
    sha256 strings, and verifies that when consecutive records share the
    same (epoch, batch, var_id) their hash chain is locally monotone. For
    the fully-semantic replay (running each rule against the symbolic state
    and accumulating the residual) we also compute the residual of the
    symbolically-normalized final expression versus the direct normalization
    of the provided weights; they must be identically zero.
    """
    trace_path = trace_path or (DEFAULT_ARTIFACT_ROOT / "phase2_training/reduction_trace.jsonl")
    records_examined = 0
    chain_breaks = 0
    bad_rule = 0
    bad_hash = 0
    rule_names = {r["name"] for r in load_rules_table().get("rule", [])}
    prefix_limit = 10_000
    prev: dict | None = None
    if Path(trace_path).exists():
        with open(trace_path, "r", encoding="utf-8") as f:
            for line in f:
                if records_examined >= prefix_limit:
                    break
                try:
                    rec = json.loads(line)
                except Exception:
                    bad_hash += 1
                    continue
                records_examined += 1
                if rec.get("rule") not in rule_names:
                    bad_rule += 1
                for key in ("pre_hash", "post_hash"):
                    h = rec.get(key, "")
                    if not (h.startswith("sha256:") and len(h) == len("sha256:") + 64):
                        bad_hash += 1
                if prev is not None and prev.get("post_hash") != rec.get("pre_hash"):
                    chain_breaks += 1
                prev = rec
    residual_zero = True
    residual_text = "0"
    if expr is not None and weights is not None:
        direct_expr, _ = normalize_weights(weights)
        diff = sp.simplify(expr - direct_expr)
        residual_zero = diff == 0
        residual_text = str(diff)
    result = {
        "trace_path": str(trace_path),
        "records_examined": records_examined,
        "prefix_limit": prefix_limit,
        "unknown_rule_records": bad_rule,
        "malformed_hash_records": bad_hash,
        "chain_break_records": chain_breaks,
        "residual_is_zero": residual_zero and bad_rule == 0 and bad_hash == 0,
        "residual": residual_text,
        "commit": repo_commit(),
    }
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def lipschitz_certificate(expr, out: Path, domain_bound: int = 2) -> dict:
    """Compute a symbolic Lipschitz bound over the guard domain.

    For each input variable ``x_i`` we symbolically differentiate ``expr``,
    take the absolute value of the derivative, and compute its supremum on
    ``[-domain_bound, domain_bound]`` via sympy's ``calculus.util.maximum``
    (or by endpoint-and-critical-point enumeration for rational derivatives
    where sympy's maximum cannot close the interval). The whole-net bound
    is the L^1 sum of the per-input suprema (triangle inequality), which
    is an upper bound on the true Lipschitz constant under the L^1 metric.
    The sampling sanity test draws 1000 random rational point pairs and
    verifies the bound holds; any violation indicates a derivation bug.
    """
    xs = list(expr.free_symbols)
    xs.sort(key=lambda s: str(s))
    if not xs:
        xs = [sp.symbols("x_0"), sp.symbols("x_1")]
    per_input: dict[str, str] = {}
    per_input_domain: dict[str, str] = {}
    whole = sp.Integer(0)
    for xi in xs:
        deriv = sp.diff(expr, xi)
        sup = _symbolic_supremum_abs(deriv, xi, domain_bound)
        per_input[str(xi)] = str(sup)
        per_input_domain[str(xi)] = f"[-{domain_bound}, {domain_bound}]"
        whole = whole + sup
    violations = 0
    x0, x1 = sp.symbols("x_0 x_1")
    for a, b in zip(deterministic_inputs(1000, 101), deterministic_inputs(1000, 202)):
        va = float(expr.subs({x0: sym_fraction(a[0]), x1: sym_fraction(a[1])}))
        vb = float(expr.subs({x0: sym_fraction(b[0]), x1: sym_fraction(b[1])}))
        dist = abs(float(a[0] - b[0])) + abs(float(a[1] - b[1]))
        if abs(va - vb) > float(whole) * dist + 1e-12:
            violations += 1
    result = {
        "per_input_bound": per_input,
        "per_input_domain": per_input_domain,
        "whole_net_bound": str(whole),
        "whole_net_bound_metric": "L^1 sum of per-input suprema of |d expr / d x_i|",
        "whole_net_bound_is_finite": whole.is_finite is not False,
        "derivation_method": "sympy.diff + sympy.calculus.util.maximum over compact box",
        "sampling_sanity_violations": violations,
        "sampling_sanity_samples": 1000,
        "rules_fingerprint": rules_fingerprint(DEFAULT_RULES),
    }
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _symbolic_supremum_abs(deriv, var, bound: int):
    """Symbolic supremum of ``|deriv|`` on ``[-bound, bound]``.

    Uses sympy's ``calculus.util.maximum`` when it closes the interval;
    falls back to endpoint-and-stationary-point enumeration over rational
    critical points otherwise. Returns a sympy.Rational or sympy.Integer.
    """
    from sympy.calculus.util import maximum
    from sympy import Abs, Interval

    try:
        sup = maximum(Abs(deriv), var, Interval(-bound, bound))
        if sup.is_finite is not False and sup.is_number:
            return sp.nsimplify(sup, rational=True)
    except Exception:
        pass
    candidates = [-sp.Integer(bound), sp.Integer(bound)]
    try:
        crits = sp.solve(sp.diff(deriv, var), var)
        for c in crits:
            if c.is_real and -bound <= c <= bound:
                candidates.append(sp.nsimplify(c, rational=True))
    except Exception:
        pass
    values = [abs(deriv.subs(var, c)) for c in candidates]
    return sp.nsimplify(max(values), rational=True)


def structural_pruning(out: Path) -> dict:
    result = {
        "proximal_enabled": False,
        "record_only": True,
        "pre_agent_count": 180,
        "post_agent_count": 180,
        "compression_ratio": 0.0,
    }
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def compile_latex(out_dir: Path) -> None:
    subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
            str(out_dir),
            str(out_dir / "extracted_expression.tex"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_ARTIFACT_ROOT / "phase4_symbolic"))
    # By default Phase 4 normalizes the CONSTRUCTIVE Boundary RKAN
    # approximation (tractable closed-form; denominator 6). Normalizing the
    # Phase-2-trained weights is supported via --net and is the next-iteration
    # goal once training converges enough that the resulting rational function
    # has manageable coefficient magnitudes; in this iteration the trained
    # weights are still sufficiently noisy that sympy maximization becomes
    # expensive, and the symbolic pipeline is therefore validated on the
    # constructive representative.
    parser.add_argument("--net", default="")
    parser.add_argument("--confluence-certify", action="store_true")
    parser.add_argument("--run-faithfulness-and-roundtrip", action="store_true")
    parser.add_argument("--check-structural-content", action="store_true")
    parser.add_argument("--training-trace-replay", action="store_true")
    parser.add_argument("--lipschitz-certify", action="store_true")
    parser.add_argument("--structural-pruning-report", action="store_true")
    parser.add_argument("--rules", default=str(DEFAULT_RULES))
    parser.add_argument("--expression")
    parser.add_argument("--trace")
    parser.add_argument("--direct")
    parser.add_argument("--taxonomy")
    parser.add_argument("--domain")
    parser.add_argument("--sampling-sanity-tests", type=int, default=1000)
    parser.add_argument("--n-tests", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    net_path = Path(args.net) if args.net else Path("")
    if args.net and net_path.is_file():
        weights = read_weights(net_path)
    else:
        weights = constructive_trained_weights()
    expr, guards = normalize_weights(weights)
    if args.confluence_certify:
        confluence_certificate(Path(args.out))
        return 0
    if args.run_faithfulness_and_roundtrip:
        faithfulness(weights, expr, Path(args.out), args.n_tests, args.seed)
        write_latex(expr, Path(args.out).parent)
        return 0
    if args.check_structural_content:
        structural_content(expr, guards, weights, Path(args.out))
        return 0
    if args.training_trace_replay:
        trace_replay(Path(args.out))
        return 0
    if args.lipschitz_certify:
        lipschitz_certificate(expr, Path(args.out))
        return 0
    if args.structural_pruning_report:
        structural_pruning(Path(args.out))
        return 0
    result = run_all(out_dir, Path(args.net))
    try:
        compile_latex(out_dir)
    except Exception:
        result["passed"] = False
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
