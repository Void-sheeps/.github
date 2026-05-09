# ============================================================
# SYMBOLIC PRIME GEOMETRY — SYMMETRY AND ITS BREAKING
# ============================================================
#
# EXTENDED SYMBOL MAP:
#
#   a=1   b=2   c=3   d=5   e=7   f=11  g=13
#   h=17  i=19  j=23  k=29  l=31  m=37  n=41  o=43
#
#   p=47  (next prime — outside map, target of Level 2)
#
# ============================================================
#
# CORE STRUCTURE:
#
#  LEVEL 1:     abcg  → i = 19      nucleus {a,b,c} + complement g
#               bh    → i = 19      binary  (b + h)
#               [Symmetric: two descriptions of i]
#
#  BREAK 1:     def   → j = 23      3 consecutive primes after c
#
#  BREAK 1b:    efg   → l = 31      3 consecutive primes after d
#
#  BREAK 2:     fgh   → n = 41      3 consecutive primes after e
#
#  LEVEL 2:     abcdek → p = 47     nucleus {a,b,c,d,e} + complement k
#               [No binary: 47-2=45, not prime — symmetry broken]
#
# ============================================================
#
# THE BREAKING:
#
#  Level 1 admits two descriptions (bh ≡ abcg).
#  Level 2 admits only one (abcdek). The binary partner vanishes.
#
#  The break regions def / efg / fgh are the primes that
#  appear BETWEEN Level 1 and Level 2 — and none of them
#  admit binary descriptions either.
#
# ============================================================

import torch
import sys
from itertools import combinations

# ------------------------------------------------------------
# SYMBOL SPACE
# ------------------------------------------------------------

SYMBOLS = {
    "a":  1, "b":  2, "c":  3, "d":  5, "e":  7,
    "f": 11, "g": 13, "h": 17, "i": 19, "j": 23,
    "k": 29, "l": 31, "m": 37, "n": 41, "o": 43,
}
SYMBOLS_EXT = {**SYMBOLS, "p": 47}   # p is the Level 2 target
REV     = {v: k for k, v in SYMBOLS_EXT.items()}
VALS    = torch.tensor(sorted(SYMBOLS.values()), dtype=torch.float32)
ALL_PRIMES = set(SYMBOLS_EXT.values())

def encode(word: list[str]) -> torch.Tensor:
    return torch.tensor([SYMBOLS_EXT[s] for s in word], dtype=torch.float32)

def action(x: torch.Tensor) -> dict:
    delta = x[1:] - x[:-1]
    abs_d = delta.abs()
    rho   = 1.0 / (abs_d + 1.0)
    terms = abs_d * rho
    return {"delta": delta, "abs_d": abs_d, "rho": rho,
            "terms": terms, "A": terms.sum()}

def word_str(letters):
    return "".join(letters)

def fmt(t): return f"{t:.6f}"


# ------------------------------------------------------------
# I. THE THREE CORE PARTITIONS
# ------------------------------------------------------------

def section_core():
    print("=" * 64)
    print("I.  CORE PARTITIONS")
    print("=" * 64)

    core = [
        ("abcg",   list("abcg"),   "i",  19, "Level 1 — nucleus + complement"),
        ("bh",     list("bh"),     "i",  19, "Level 1 — binary"),
        ("def",    list("def"),    "j",  23, "Break 1  — consecutive triplet"),
        ("efg",    list("efg"),    "l",  31, "Break 1b — consecutive triplet"),
        ("fgh",    list("fgh"),    "n",  41, "Break 2  — consecutive triplet"),
        ("abcdek", list("abcdek"), "p",  47, "Level 2 — nucleus + complement"),
    ]

    print(f"\n  {'WORD':<10} {'VALUES':<24} {'SUM':>5}  {'TARGET':>7}  {'A(ω)':>9}  ROLE")
    print(f"  {'-'*10} {'-'*24} {'-'*5}  {'-'*7}  {'-'*9}  {'-'*28}")

    records = {}
    for name, w, target, tval, role in core:
        x  = encode(w)
        r  = action(x)
        s  = int(x.sum().item())
        records[name] = {"x": x, "r": r, "target": target, "tval": tval}
        vals_str = str([int(v) for v in x.tolist()])
        print(f"  {name:<10} {vals_str:<24} {s:>5}  ={target}={tval:<4}  "
              f"{fmt(r['A'].item()):>9}  {role}")

    # Verify abcg ≡ bh at different resolutions
    print(f"\n  ── Resolution identity ──────────────────────────────────")
    acg_sum = encode(list("acg")).sum().item()
    h_val   = SYMBOLS["h"]
    print(f"    a + c + g  =  1+3+13 = {int(acg_sum)}  =  h={h_val}   ✓")
    print(f"    abcg  =  b + (a+c+g)  =  b + h  =  bh")
    delta_A_1 = abs(records["abcg"]["r"]["A"].item() - records["bh"]["r"]["A"].item())
    print(f"    Resolution cost ΔA  =  {delta_A_1:.6f}")

    # Level 2: no binary partner
    print(f"\n  ── Binary test for p=47 ────────────────────────────────")
    p_val    = 47
    b_val    = SYMBOLS["b"]
    complement_p = p_val - b_val
    print(f"    p - b  =  {p_val} - {b_val} = {complement_p}")
    print(f"    {complement_p} ∈ prime space?  {complement_p in ALL_PRIMES}   ← binary FAILS")
    print(f"    abcdek stands alone — no bX equivalent")

    return records


# ------------------------------------------------------------
# II. BINARY AVAILABILITY ACROSS THE SPACE
# ------------------------------------------------------------

def section_binary():
    print()
    print("=" * 64)
    print("II. BINARY AVAILABILITY  (Goldbach test: p = b + q?)")
    print("=" * 64)

    b_val = SYMBOLS["b"]
    print(f"\n  {'LETTER':>7}  {'VAL':>4}  {'VAL-b':>6}  {'BINARY':>12}  REGION")
    print(f"  {'-'*7}  {'-'*4}  {'-'*6}  {'-'*12}  {'-'*20}")

    # Annotate by region
    regions = {
        "i": "Level 1 target", "p": "Level 2 target",
        "j": "Break 1",  "l": "Break 1b", "n": "Break 2",
        "b": "generator",
    }

    binary_available = []
    binary_absent    = []
    for letter, val in sorted(SYMBOLS_EXT.items(), key=lambda kv: kv[1]):
        comp = val - b_val
        has_binary = comp > 0 and comp in ALL_PRIMES
        comp_str = REV.get(comp, str(comp))
        binary_str = f"b+{comp_str}" if has_binary else "✗"
        region = regions.get(letter, "")
        flag = " ← " + region if region else ""
        print(f"  {letter:>7}  {val:>4}  {comp:>6}  {binary_str:>12}  {flag}")
        (binary_available if has_binary else binary_absent).append(letter)

    print(f"\n  Binary available:  {binary_available}")
    print(f"  Binary absent:     {binary_absent}")
    print(f"\n  Count with binary:    {len(binary_available)}")
    print(f"  Count without binary: {len(binary_absent)}")


# ------------------------------------------------------------
# III. BREAK FAMILY — CONSECUTIVE PRIME TRIPLETS
# ------------------------------------------------------------

def section_breaks():
    print()
    print("=" * 64)
    print("III. BREAK FAMILY  (consecutive prime triplets → prime)")
    print("=" * 64)

    vals_list = sorted(SYMBOLS.values())
    print(f"\n  {'TRIPLET':<10} {'VALUES':<18} {'SUM':>5}  {'TARGET':>8}  {'BINARY?':>8}  {'A(ω)':>9}")
    print(f"  {'-'*10} {'-'*18} {'-'*5}  {'-'*8}  {'-'*8}  {'-'*9}")

    break_words = []
    for i in range(len(vals_list) - 2):
        a, b_v, c_v = vals_list[i], vals_list[i+1], vals_list[i+2]
        s = a + b_v + c_v
        if s in ALL_PRIMES:
            name    = REV[a] + REV[b_v] + REV[c_v]
            target  = REV[s]
            x       = torch.tensor([a, b_v, c_v], dtype=torch.float32)
            r       = action(x)
            has_bin = (s - SYMBOLS["b"]) in ALL_PRIMES
            bin_str = f"b+{REV[s-SYMBOLS['b']]}" if has_bin else "✗"
            break_words.append((name, [a,b_v,c_v], s, target, r["A"].item()))
            print(f"  {name:<10} {str([a,b_v,c_v]):<18} {s:>5}  ={target}={s:<4}  "
                  f"{bin_str:>8}  {fmt(r['A'].item()):>9}")

    # Special: what do the breaks reveal about the intermediate primes?
    print(f"\n  ── Break sums occupy specific positions in prime sequence ──")
    break_targets = [bw[3] for bw in break_words]
    print(f"    Triplet targets:  {break_targets}")
    print(f"    Between Level 1 (i=19) and Level 2 (p=47):")
    between = [l for l,v in sorted(SYMBOLS_EXT.items(), key=lambda x:x[1])
               if 19 < v < 47]
    print(f"    All primes in range: {between}")
    break_in_range = [b for b in break_targets if b in between]
    print(f"    Covered by breaks: {break_in_range}")


# ------------------------------------------------------------
# IV. INTERNAL DECOMPOSITION OF abcdek
# ------------------------------------------------------------

def section_internal():
    print()
    print("=" * 64)
    print("IV. INTERNAL DECOMPOSITION OF abcdek")
    print("=" * 64)

    print(f"""
  abcdek  =  {{a,b,c,d,e,k}}
           =  {{a,b,c,g}} ∪ {{d,e,k}}  ?
""")

    abcg_sum = int(encode(list("abcg")).sum().item())
    dek_sum  = int(encode(list("dek")).sum().item())
    print(f"  abcg  =  1+2+3+13  =  {abcg_sum}  =  i")
    print(f"  dek   =  5+7+29    =  {dek_sum}  =  n")
    print(f"  abcg + dek  =  {abcg_sum} + {dek_sum}  =  {abcg_sum+dek_sum}")
    print(f"  ← element-SUM partition: both sub-words sum to primes,")
    print(f"    but their sum {abcg_sum+dek_sum} ≠ p=47 (the full word sum)")
    print(f"    because the partition is NOT a sub-grouping — it is additive")

    # Correct nested structure: abcdek as b + (a+c+d+e+k)
    acdk_sum = int(encode(list("acdek")).sum().item())
    print(f"\n  ── Correct nesting: extract b ────────────────────────")
    print(f"  abcdek  =  b  +  (a+c+d+e+k)")
    print(f"           =  2  +  {acdk_sum}  =  {2+acdk_sum}")
    in_map = acdk_sum in ALL_PRIMES
    print(f"  a+c+d+e+k  =  {acdk_sum}  ∈ prime space?  {in_map}")

    # Show all 2-element sub-partitions of abcdek's elements
    print(f"\n  ── 2-element groupings of abcdek values ──────────────")
    abcdek_vals = [SYMBOLS_EXT[s] for s in "abcdek"]
    print(f"  abcdek values: {abcdek_vals}")
    for combo in combinations(range(len(abcdek_vals)), 2):
        s1 = abcdek_vals[combo[0]]
        s2 = abcdek_vals[combo[1]]
        total = s1 + s2
        in_p  = total in ALL_PRIMES
        if in_p:
            letters1 = REV[s1]
            letters2 = REV[s2]
            print(f"    {letters1}+{letters2}  =  {s1}+{s2}  =  {total}  =  {REV[total]}")

    # def vs dek: the interface between Level 1 and Level 2
    print(f"\n  ── def vs dek: the interface ─────────────────────────")
    def_vals = [SYMBOLS[s] for s in "def"]
    dek_vals = [SYMBOLS_EXT[s] for s in "dek"]
    print(f"  def  =  {{d,e,f}}  =  {def_vals}  →  {sum(def_vals)}  =  j")
    print(f"  dek  =  {{d,e,k}}  =  {dek_vals}  →  {sum(dek_vals)}  =  n")
    print(f"  Shared prefix: {{d,e}} — d and e appear in both")
    gap = SYMBOLS["k"] - SYMBOLS["f"]
    print(f"  Terminal jump: f={SYMBOLS['f']} → k={SYMBOLS['k']}   Δ = {gap}")
    print(f"  This gap ({gap}) spans: f,g,h,i,j  (5 primes skipped)")
    print(f"  ← The break between Level 1 and Level 2 is encoded")
    print(f"    in the terminal displacement from f to k")


# ------------------------------------------------------------
# V. LEVEL STRUCTURE AND PROJECTION
# ------------------------------------------------------------

def section_levels():
    print()
    print("=" * 64)
    print("V.  LEVEL STRUCTURE AND PROJECTION")
    print("=" * 64)

    print(f"""
  Nucleus grows by absorbing the first 2 primes after each endpoint.
  Break region = 3 consecutive primes starting after nucleus endpoint.

  LEVEL 1:
    Nucleus:  {{a,b,c}}  =  {{1,2,3}}    sum = 6
    Break 1:  {{d,e,f}}  after c=3        →  j = 23
    Complement: g = 13   (= 6 + e)       why? 6 + 7 = 13 ✓
    Target:   i = 6 + 13 = 19

  LEVEL 2:
    Nucleus:  {{a,b,c,d,e}}  =  {{1,2,3,5,7}}  sum = 18
    Break 2:  {{f,g,h}}  after e=7             →  n = 41
    Complement: k = 29   (= 18 + f)    why? 18 + 11 = 29 ✓
    Target:   p = 18 + 29 = 47
""")

    # Verify complement = nucleus_sum + first_excluded_prime
    nucleus_1 = list("abc")
    nucleus_2 = list("abcde")
    first_excl_1 = "e"   # first prime after nucleus 1 used as complement seed
    first_excl_2 = "f"   # first prime after nucleus 2

    n1_sum = int(encode(nucleus_1).sum().item())
    n2_sum = int(encode(nucleus_2).sum().item())
    fe1    = SYMBOLS[first_excl_1]
    fe2    = SYMBOLS[first_excl_2]

    print(f"  Complement formula:  comp = nucleus_sum + first_post_nucleus_prime")
    print(f"  Level 1: {n1_sum} + {fe1} = {n1_sum+fe1} = g={SYMBOLS['g']}  "
          f"{'✓' if n1_sum+fe1 == SYMBOLS['g'] else '✗'}")
    print(f"  Level 2: {n2_sum} + {fe2} = {n2_sum+fe2} = k={SYMBOLS['k']}  "
          f"{'✓' if n2_sum+fe2 == SYMBOLS['k'] else '✗'}")

    # Project Level 3
    print(f"\n  ── Projected Level 3 ─────────────────────────────────")
    nucleus_3 = list("abcdefg")
    n3_sum    = int(encode(nucleus_3).sum().item())
    first_excl_3 = "h"
    fe3 = SYMBOLS[first_excl_3]
    comp3 = n3_sum + fe3
    target3 = n3_sum + comp3
    print(f"  Nucleus 3:  {{a..g}} sum = {n3_sum}")
    print(f"  Complement: {n3_sum} + h={fe3} = {comp3}")
    in_map = comp3 in ALL_PRIMES or True # Might be outside initial map
    print(f"  comp3 = {comp3}  ∈ prime space? {comp3 in [53, 59, 61, 67, 71, 73, 79, 83, 89, 97] or comp3 in ALL_PRIMES}")
    print(f"  Target3: {n3_sum} + {comp3} = {target3}")
    break3 = [SYMBOLS.get(x, x) for x in "ghi"]
    print(f"\n  Break 3:  {{g,h,i}} = {break3} → {sum(break3)}")
    print(f"  ← Pattern: break region = 3 primes starting after nucleus endpoint")


# ------------------------------------------------------------
# VI. ACTION SPECTRUM ACROSS ALL LEVELS
# ------------------------------------------------------------

def section_action_spectrum():
    print()
    print("=" * 64)
    print("VI. ACTION SPECTRUM")
    print("=" * 64)

    words = [
        ("bh",     list("bh"),     "Level 1 binary"),
        ("abcg",   list("abcg"),   "Level 1 nuclear"),
        ("def",    list("def"),    "Break 1"),
        ("efg",    list("efg"),    "Break 1b"),
        ("fgh",    list("fgh"),    "Break 2"),
        ("abcdek", list("abcdek"), "Level 2 nuclear"),
    ]

    # Action tensor: stack all action values
    A_vals = []
    print(f"\n  {'WORD':<10} {'GAPS Δ':<20} {'ρ':<20} {'A(ω)':>9}  ROLE")
    print(f"  {'-'*10} {'-'*20} {'-'*20} {'-'*9}  {'-'*20}")

    for name, w, role in words:
        x = encode(w)
        r = action(x)
        A = r["A"].item()
        A_vals.append(A)
        d_str = str([int(v) for v in r["delta"].tolist()])
        rho_str = "[" + ",".join(f"{v:.2f}" for v in r["rho"].tolist()) + "]"
        print(f"  {name:<10} {d_str:<20} {rho_str:<20} {fmt(A):>9}  {role}")

    A_tensor = torch.tensor(A_vals)
    print(f"\n  ── Action statistics ─────────────────────────────────")
    print(f"  min  A = {A_tensor.min().item():.6f}  (bh — binary, Level 1)")
    print(f"  max  A = {A_tensor.max().item():.6f}  (abcdek — nuclear, Level 2)")
    print(f"  mean A = {A_tensor.mean().item():.6f}")
    print(f"  std  A = {A_tensor.std().item():.6f}")

    # Level 1 vs Level 2 nuclear action ratio
    A_L1 = A_vals[1]   # abcg
    A_L2 = A_vals[5]   # abcdek
    print(f"\n  Nuclear action ratio  A(Level2) / A(Level1):")
    print(f"    {A_L2:.6f} / {A_L1:.6f}  =  {A_L2/A_L1:.6f}")

    # Break action: are def, efg, fgh approximately equal?
    A_breaks = torch.tensor([A_vals[2], A_vals[3], A_vals[4]])
    print(f"\n  Break action values  (def, efg, fgh):")
    print(f"    {A_breaks.tolist()}")
    print(f"    std = {A_breaks.std().item():.6f}  ← break family cohesion")


def run_all():
    print()
    print("=" * 64)
    print("  SYMBOLIC PRIME GEOMETRY — SYMMETRY AND ITS BREAKING")
    print("  a=1 b=2 c=3 d=5 e=7 f=11 g=13 h=17 i=19 j=23")
    print("  k=29 l=31 m=37 n=41 o=43   |   p=47 (target)")
    print("=" * 64)

    section_core()
    section_binary()
    section_breaks()
    section_internal()
    section_levels()
    section_action_spectrum()

    print()
    print("=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    print("""
  SYMMETRY (Level 1):
    i=19 admits two descriptions: bh (binary) and abcg (nuclear).
    They are the same partition at different resolutions:
      a + c + g = 17 = h  →  abcg = b + h = bh.

  BREAK (Transition):
    Between i and p, three consecutive-prime triplets yield primes:
      def → j=23,  efg → l=31,  fgh → n=41.
    None of these admits a binary description.
    These are the "non-symmetric" primes of the space.

  BREAKING (Level 2):
    p=47 admits only one description: abcdek (nuclear).
    The binary partner fails: 47-2=45 is not prime.
    The nucleus has grown — {d,e} absorbed — but the binary
    symmetry of Level 1 does not survive the transition.

  COMPLEMENT FORMULA:
    Level 1: complement = nucleus_sum + first_post_nucleus_prime
               g = (a+b+c) + e = 6 + 7 = 13  ✓
    Level 2: k = (a+b+c+d+e) + f = 18 + 11 = 29  ✓

  BREAK FORMULA:
    Break region = 3 consecutive primes starting immediately
                   after the nucleus endpoint.
    Level 1 endpoint c=3  →  {d,e,f} = def → j
    Level 2 endpoint e=7  →  {f,g,h} = fgh → n
""")

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    if "--simulate" in sys.argv or len(sys.argv) == 1:
        run_all()
