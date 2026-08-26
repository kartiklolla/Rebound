#!/usr/bin/env bash
# Break one thing on purpose, and confirm the suite notices.
#
#     ./scripts/mutation_check.sh
#
# A passing test suite says the code does what the tests check. It says nothing
# about whether the tests check anything. These mutations disable a guarantee
# the project claims in prose; each one must turn the suite red.
#
# Committed because an earlier, larger runner was not. It lived in a scratch
# directory that was wiped, and the README went on citing "27 mutations, 27
# caught" — a number nobody outside this machine could reproduce, held to a
# looser standard than the two claims the README withdraws for exactly that
# reason. This is smaller and it runs.
#
# A mutation that does not change the file is refused rather than scored: perl
# exits 0 when its pattern misses, so a stale mutation silently reads as a hole
# in the tests rather than a hole in this script. That happened, four times at
# once, and the diff guard below is why it cannot happen quietly again.
set -u
SRC="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$SRC/.venv/bin/python}"
SUITE="${SUITE:-tests}"
caught=0; survived=0; noop=0

run_mutation () {
  local name="$1"; shift
  local work; work="$(mktemp -d)"
  cp -r "$SRC/src" "$SRC/tests" "$SRC/pyproject.toml" "$work/"
  ( cd "$work" && "$@" ) || { echo "  [$name] MUTATION ERRORED"; rm -rf "$work"; return; }

  if diff -rq "$SRC/src" "$work/src" >/dev/null 2>&1; then
    echo "  [$name] NO-OP — pattern did not match; rewrite the mutation"
    noop=$((noop+1)); rm -rf "$work"; return
  fi

  local out
  out=$( cd "$work" && "$PYTHON" -m pytest "$SUITE" -q -x 2>&1 | tail -1 )
  if echo "$out" | grep -q "failed\|error"; then
    echo "  [$name] caught"; caught=$((caught+1))
  else
    echo "  [$name] SURVIVED — $out"; survived=$((survived+1))
  fi
  rm -rf "$work"
}

echo "Mutating the source. Every line below must say 'caught'."
echo

# --- the verifier's guarantees ---------------------------------------------
run_mutation "amount check disabled" \
  perl -0pi -e 's/(class AmountIsExact:.*?def run\(self, draft: Draft, brief: MessageBrief\) -> Finding \| None:\n)/$1        return None\n/s' src/rebound/verify.py

run_mutation "link check disabled" \
  perl -0pi -e 's/(class LinksAreOurs:.*?def run\(self, draft: Draft, brief: MessageBrief\) -> Finding \| None:\n)/$1        return None\n/s' src/rebound/verify.py

run_mutation "instruction check disabled" \
  perl -0pi -e 's/(class AskIsHonoured:.*?def run\(self, draft: Draft, brief: MessageBrief\) -> Finding \| None:\n)/$1        return None\n/s' src/rebound/verify.py

run_mutation "bare hosts stop counting as links" \
  perl -0pi -e 's/    return _DOT_THEN_LETTER\.search\(stripped\) is not None/    return False/s' src/rebound/verify.py

run_mutation "coercion stems dropped" \
  perl -0pi -e 's/^_COERCION_STEMS: tuple\[str, \.\.\.\] = \(/_COERCION_STEMS: tuple[str, ...] = ()\nUNUSED = (/m' src/rebound/verify.py

# --- the desk's guarantees --------------------------------------------------
run_mutation "desk skips verification" \
  perl -0pi -e 's/        return Attempt\(draft=draft, findings=verify\(draft, brief\)\)/        return Attempt(draft=draft, findings=())/s' src/rebound/desk.py

run_mutation "fallback removed" \
  perl -0pi -e 's/        final = self\._attempt\(self\.fallback, brief\)/        final = self._attempt(self.drafter, brief)/s' src/rebound/desk.py

run_mutation "repair feedback dropped" \
  perl -0pi -e 's/            previous, feedback = attempt\.draft, feedback_for\(attempt\.findings\)/            previous, feedback = None, None/s' src/rebound/desk.py

# --- the brief's guarantees -------------------------------------------------
run_mutation "approval type check removed" \
  perl -0pi -e 's/        if not isinstance\(approved, ApprovedAction\):/        if False:/s' src/rebound/comms.py

run_mutation "structural legality check removed" \
  perl -0pi -e 's/        if failure_code is not None and action not in legal_actions\(failure_code\):/        if False:/s' src/rebound/comms.py

run_mutation "SMS segment budget inflated" \
  perl -pi -e 's/if units <= 70:/if units <= 700:/' src/rebound/comms.py

# --- the compliance gate ----------------------------------------------------
run_mutation "contact suppression ignored" \
  perl -0pi -e 's/        if not request\.contact_suppressed:\n            return None\n/        return None\n/s' src/rebound/compliance.py

run_mutation "retry cap off by one again" \
  perl -0pi -e 's/        presentations = MAX_EXECUTIONS_PER_CYCLE \+ request\.debit_attempts/        presentations = request.debit_attempts/s' src/rebound/compliance.py

# --- the harness ------------------------------------------------------------
run_mutation "draws go back to one shared stream" \
  perl -0pi -e 's/        if episode\.entropy is None:\n            return float\(self\.rng\.random\(\)\)/        if True:\n            return float(self.rng.random())/s' src/rebound/sim/world.py

run_mutation "episode id reveals its stream index" \
  perl -0pi -e 's/    return f"EV_\{digest\}"/    return f"EV_{index:08d}"/s' src/rebound/eval/harness.py

# --- the model --------------------------------------------------------------
run_mutation "group-atomic split falls through" \
  perl -0pi -e 's/            raise ValueError\(/            pass if False else None\n            _unused = ValueError(/s' src/rebound/model.py

echo
echo "caught $caught · survived $survived · no-op $noop"
[ "$survived" -eq 0 ] && [ "$noop" -eq 0 ]
