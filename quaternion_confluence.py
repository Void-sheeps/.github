import argparse
import itertools
from dataclasses import dataclass
from itertools import count
from functools import cmp_to_key
from typing import List, Tuple, Dict, Set, Callable, Optional


# ---------------------------------------------------------------------------
# Core Rewriting Objects & Helpers
# ---------------------------------------------------------------------------

def inv(letter: str) -> str:
    """Formal inverse: lowercase <-> uppercase."""
    return letter.lower() if letter.isupper() else letter.upper()


def free_reduce(w: tuple) -> tuple:
    """Cancel adjacent x, x^-1 pairs automatically (free group reduction)."""
    stack = []
    for c in w:
        if stack and stack[-1] == inv(c):
            stack.pop()
        else:
            stack.append(c)
    return tuple(stack)


def make_shortlex(letter_order: list) -> Callable[[tuple, tuple], int]:
    """Returns a comparison function cmp(u, v) -> -1/0/1 giving a total,
    well-founded order compatible with concatenation (needed for termination)."""
    rank = {c: i for i, c in enumerate(letter_order)}

    def cmp(u: tuple, v: tuple) -> int:
        if len(u) != len(v):
            return -1 if len(u) < len(v) else 1
        for a, b in zip(u, v):
            if a != b:
                ra, rb = rank[a], rank[b]
                return -1 if ra < rb else 1
        return 0
    return cmp


@dataclass
class Rule:
    lhs: tuple
    rhs: tuple
    origin: str

    def __repr__(self) -> str:
        l = ''.join(self.lhs) or 'e'
        r = ''.join(self.rhs) or 'e'
        return f"{l} -> {r}"

    def __str__(self) -> str:
        return self.__repr__()


def rewrite_step(w: tuple, rules: List[Rule]) -> Tuple[Optional[tuple], Optional[Rule]]:
    """Apply the first matching rule anywhere in w; return new word or None."""
    n = len(w)
    for rule in rules:
        m = len(rule.lhs)
        if m == 0:
            continue
        for i in range(n - m + 1):
            if w[i:i + m] == rule.lhs:
                return free_reduce(w[:i] + rule.rhs + w[i + m:]), rule
    return None, None


def normal_form(w: tuple, rules: List[Rule], max_steps: int = 2000) -> Tuple[tuple, List[Rule]]:
    """Reduce w to its unique rules-normal form."""
    w = free_reduce(w)
    trace = []
    for _ in range(max_steps):
        result = rewrite_step(w, rules)
        if result[0] is None:
            return w, trace
        w, used = result
        trace.append(used)
    raise RuntimeError(f"normal_form did not terminate within {max_steps} steps on {w}")


# ---------------------------------------------------------------------------
# Critical Pair Analysis (Overlaps)
# ---------------------------------------------------------------------------

def overlaps(l1: tuple, l2: tuple):
    """Yield all suffix-prefix and inclusion overlaps of l1 with l2."""
    n1, n2 = len(l1), len(l2)
    # Suffix-prefix overlaps
    for i in range(1, n1):
        k = n1 - i
        if k <= n2 and l1[i:] == l2[:k]:
            yield l1 + l2[k:], 0, i
    # Inclusions
    for i in range(0, n1 - n2 + 1):
        if n2 > 0 and l1[i:i + n2] == l2:
            yield l1, 0, i


def critical_pairs(rules: List[Rule]) -> List[Tuple[tuple, tuple, str]]:
    """Generate all critical pairs that must be resolved to ensure confluence."""
    pairs = []
    for r1 in rules:
        for r2 in rules:
            for amb, p1, p2 in overlaps(r1.lhs, r2.lhs):
                s = free_reduce(amb[:p1] + r1.rhs + amb[p1 + len(r1.lhs):])
                t = free_reduce(amb[:p2] + r2.rhs + amb[p2 + len(r2.lhs):])
                desc = f"overlap({r1}, {r2}) on {''.join(amb) or 'e'}"
                pairs.append((s, t, desc))
    return pairs


# ---------------------------------------------------------------------------
# Completion Engine
# ---------------------------------------------------------------------------

class CompletionFailure(Exception):
    pass


def knuth_bendix(rules: List[Rule], cmp: Callable[[tuple, tuple], int], max_iters: int = 200, verbose: bool = True) -> List[Rule]:
    """Execute the Knuth-Bendix completion algorithm to find a confluent rule-set."""
    rules = list(rules)
    counter = count(1)
    for iteration in range(max_iters):
        pairs = critical_pairs(rules)
        new_rule_added = False
        for s, t, desc in pairs:
            ns, _ = normal_form(s, rules)
            nt, _ = normal_form(t, rules)
            if ns == nt:
                continue  # already joinable
            c = cmp(ns, nt)
            if c == 0:
                continue
            lhs, rhs = (ns, nt) if c > 0 else (nt, ns)
            new_rule = Rule(lhs, rhs, origin=f"CP #{next(counter)}: {desc}")
            rules.append(new_rule)
            if verbose:
                print(f"  [iter {iteration:2d}] New Rule: {str(new_rule):<12} (From {desc})")
            new_rule_added = True
        if not new_rule_added:
            if verbose:
                print(f"\nConfluence achieved after {iteration} iterations. Total Rules: {len(rules)}.")
            return rules
    raise CompletionFailure("Failed to converge (infinite system or weak ordering).")


# ---------------------------------------------------------------------------
# Mathematical Exploration of the Completed Group
# ---------------------------------------------------------------------------

def discover_group_elements(generators: list, rules: List[Rule], cmp_fn: Callable[[tuple, tuple], int]) -> List[tuple]:
    """Breadth-First Search on the Cayley graph to dynamically discover all
    unique group elements in their canonical normal forms."""
    all_gens = list(generators) + [inv(g) for g in generators]
    visited = {()}
    queue = [()]
    while queue:
        curr = queue.pop(0)
        for g in all_gens:
            neighbor, _ = normal_form(curr + (g,), rules)
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return sorted(list(visited), key=cmp_to_key(cmp_fn))


def format_word(w: tuple) -> str:
    """Format tuples into beautiful algebraic math notation (e.g., 'a^-1', 'a^2b')."""
    if not w:
        return '1'
    res = []
    i = 0
    while i < len(w):
        char = w[i]
        display_char = f"{char.lower()}^-1" if char.isupper() else char
        count_val = 1
        while i + 1 < len(w) and w[i+1] == char:
            count_val += 1
            i += 1
        if count_val > 1:
            if char.isupper():
                res.append(f"{char.lower()}^-{count_val}")
            else:
                res.append(f"{char}^{count_val}")
        else:
            res.append(display_char)
        i += 1
    return "".join(res)


def print_cayley_table(elements: List[tuple], mult: Dict[Tuple[tuple, tuple], tuple]) -> None:
    """Prints a perfectly aligned ASCII multiplication table for the group."""
    names = {el: format_word(el) for el in elements}
    max_len = max(len(names[el]) for el in elements)
    header = f"{'':>{max_len}} | " + " | ".join(f"{names[el]:^{max_len}}" for el in elements)
    print(header)
    print("-" * len(header))
    for row in elements:
        row_str = f"{names[row]:>{max_len}} | "
        row_str += " | ".join(f"{names[mult[(row, col)]]:^{max_len}}" for col in elements)
        print(row_str)


# ---------------------------------------------------------------------------
# Execution Pipeline
# ---------------------------------------------------------------------------

def run_simulation() -> None:
    print("=========================================================")
    # 1. Setup Alphabet and Shortlex Ordering
    # Generators before inverses: 'a' < 'b' < 'A' < 'B'
    letter_order = ['a', 'b', 'A', 'B']
    cmp_fn = make_shortlex(letter_order)

    # 2. Input the defining relators of Q_8
    r0 = Rule(('a', 'a', 'a', 'a'), (), "a^4 = 1")
    r1 = Rule(('b', 'b'), ('a', 'a'), "b^2 = a^2")
    r2 = Rule(('a', 'b', 'a'), ('b',), "aba = b")

    # Explicit rules modeling the free group inverses (so KB can overlap them)
    inv_rules = [
        Rule(('a', 'A'), (), "a A = 1"),
        Rule(('A', 'a'), (), "A a = 1"),
        Rule(('b', 'B'), (), "b B = 1"),
        Rule(('B', 'b'), (), "B b = 1"),
    ]

    print("--- Phase 1: Initiating Knuth-Bendix Completion on Q_8 ---")
    initial_rules = [r0, r1, r2] + inv_rules
    final_rules = knuth_bendix(initial_rules, cmp_fn, verbose=True)

    # 3. Discover Group Elements
    print("\n--- Phase 2: Dynamical Group Element Enumeration (BFS) ---")
    elements = discover_group_elements(['a', 'b'], final_rules, cmp_fn)
    print(f"Successfully discovered {len(elements)} unique elements of Q_8:")
    for el in elements:
        print(f"  Normal Form: {str(el):15s} => Algebraically: {format_word(el)}")

    # 4. Generate Cayley Table
    print("\n--- Phase 3: Generating Cayley Multiplication Table ---")
    mult = {}
    inv_map = {}
    for u in elements:
        for v in elements:
            prod, _ = normal_form(u + v, final_rules)
            mult[(u, v)] = prod
            if prod == ():
                inv_map[u] = v

    print_cayley_table(elements, mult)

    # 5. Verify the Hamiltonian Property (All Subgroups are Normal)
    print("\n--- Phase 4: Proving the Hamiltonian Property of Q_8 ---")
    # Identify all subgroups
    subgroups = []
    for r in range(1, len(elements) + 1):
        for subset in itertools.combinations(elements, r):
            if () not in subset:
                continue
            # Check closure
            closed = True
            for x in subset:
                for y in subset:
                    if mult[(x, y)] not in subset:
                        closed = False
                        break
                if not closed:
                    break
            if closed:
                subgroups.append(set(subset))

    print(f"Total subgroups found: {len(subgroups)}")
    all_subgroups_normal = True

    for idx, H in enumerate(subgroups, 1):
        is_normal = True
        for g in elements:
            g_inv = inv_map[g]
            for h in H:
                gh = mult[(g, h)]
                ghg_inv = mult[(gh, g_inv)]
                if ghg_inv not in H:
                    is_normal = False
                    break
            if not is_normal:
                break

        h_formatted = ", ".join(format_word(h) for h in sorted(H, key=cmp_to_key(cmp_fn)))
        status = "NORMAL (gHg^-1 ⊆ H)" if is_normal else "FAILED"
        print(f"  Subgroup H_{idx} (Size {len(H):2d}): {{{h_formatted}}} -> {status}")
        if not is_normal:
            all_subgroups_normal = False

    if all_subgroups_normal:
        print("\n[PROVEN] Every subgroup of Q_8 is normal! Q_8 is indeed a Hamiltonian group.")
    print("=========================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knuth-Bendix Completion & Quaternion Group Exploration")
    parser.add_argument("--simulate", action="store_true", help="Execute the complete pipeline")
    args = parser.parse_args()

    if args.simulate or not any(vars(args).values()):
        run_simulation()
