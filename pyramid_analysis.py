from collections import deque
from fractions import Fraction
from itertools import combinations, combinations_with_replacement
from random import Random


N = 64
LEVEL_SIZES = (2, 2, 4, 8, 16, 32)
LEVEL_ENDS = (2, 4, 8, 16, 32, 64)
MAX_FREE_VARIABLES = 16
RANDOM_SEED = 42


def pyramid(word):
    if len(word) != N:
        raise ValueError(f"Expected {N} elements.")

    current = list(word)
    details = []

    while len(current) > 1:
        next_sums = [
            current[i] + current[i + 1]
            for i in range(0, len(current), 2)
        ]

        differences = [
            current[i] - current[i + 1]
            for i in range(0, len(current), 2)
        ]

        details.append(differences)
        current = next_sums

    return current + [
        value
        for level in reversed(details)
        for value in level
    ]


def level_of(index):
    if not 0 <= index < N:
        raise IndexError(index)

    for level, end in enumerate(LEVEL_ENDS):
        if index < end:
            return level

    raise IndexError(index)


def transformation_matrix():
    columns = []

    for index in range(N):
        basis = [0] * N
        basis[index] = 1
        columns.append(pyramid(basis))

    return [
        [columns[column][row] for column in range(N)]
        for row in range(N)
    ]


def symmetry_operator(word, start, width):
    if len(word) != N:
        raise ValueError(f"Expected {N} elements.")

    if width < 2 or width & (width - 1):
        raise ValueError("Width must be a power of two greater than one.")

    if start < 0 or start + width > N or start % width:
        raise ValueError("Invalid aligned block.")

    result = list(word)
    half = width // 2

    left = result[start:start + half]
    right = result[start + half:start + width]

    result[start:start + width] = right + left

    return tuple(result)


def symmetry_key(word):
    if len(word) != N:
        raise ValueError(f"Expected {N} elements.")

    def canonical(start, width):
        if width == 1:
            return word[start]

        half = width // 2
        left = canonical(start, half)
        right = canonical(start + half, half)

        return tuple(sorted((left, right)))

    return canonical(0, N)


def symmetry_orbit(word, limit=10000):
    initial = tuple(word)

    if len(initial) != N:
        raise ValueError(f"Expected {N} elements.")

    visited = {initial}
    queue = deque([initial])

    widths = [
        1 << level
        for level in range(1, N.bit_length())
    ]

    while queue and len(visited) < limit:
        current = queue.popleft()

        for width in widths:
            for start in range(0, N, width):
                candidate = symmetry_operator(
                    current,
                    start,
                    width
                )

                if candidate not in visited:
                    visited.add(candidate)
                    queue.append(candidate)

                    if len(visited) >= limit:
                        break

            if len(visited) >= limit:
                break

    return visited


def symmetry_classes(words):
    classes = {}

    for word in words:
        key = symmetry_key(word)
        classes.setdefault(key, []).append(tuple(word))

    return classes


def gauss_jordan(rows, variable_count):
    matrix = [
        [Fraction(value) for value in row]
        for row in rows
    ]

    row_count = len(matrix)
    pivot_columns = []
    pivot_row = 0

    for column in range(variable_count):
        selected = next(
            (
                row
                for row in range(pivot_row, row_count)
                if matrix[row][column] != 0
            ),
            None
        )

        if selected is None:
            continue

        matrix[pivot_row], matrix[selected] = (
            matrix[selected],
            matrix[pivot_row]
        )

        pivot = matrix[pivot_row][column]

        matrix[pivot_row] = [
            value / pivot
            for value in matrix[pivot_row]
        ]

        for row in range(row_count):
            if row == pivot_row:
                continue

            factor = matrix[row][column]

            if factor == 0:
                continue

            matrix[row] = [
                matrix[row][j] - factor * matrix[pivot_row][j]
                for j in range(variable_count + 1)
            ]

        pivot_columns.append(column)
        pivot_row += 1

        if pivot_row == row_count:
            break

    for row in matrix:
        if (
            all(row[column] == 0 for column in range(variable_count))
            and row[variable_count] != 0
        ):
            return None, None

    return matrix, pivot_columns


def reconstruct(
    anchors,
    coefficients,
    matrix,
    max_free_variables=MAX_FREE_VARIABLES
):
    rows = [
        list(matrix[index]) + [Fraction(value)]
        for index, value in anchors.items()
    ]

    reduced, pivots = gauss_jordan(rows, N)

    if reduced is None:
        return []

    pivot_set = set(pivots)
    free_columns = [
        column
        for column in range(N)
        if column not in pivot_set
    ]

    if len(free_columns) > max_free_variables:
        raise ValueError(
            f"Reconstruction requires {len(free_columns)} free variables. "
            f"Limit is {max_free_variables}."
        )

    solutions = []

    reduced_float = [[float(val) for val in row] for row in reduced]
    free_count = len(free_columns)
    pivot_count = len(pivots)

    for mask in range(1 << free_count):
        free_vals = [(mask >> bit) & 1 for bit in range(free_count)]

        valid = True
        pivot_vals = []

        for row_index in range(pivot_count):
            val = reduced_float[row_index][N]
            for bit, column in enumerate(free_columns):
                if free_vals[bit]:
                    val -= reduced_float[row_index][column]

            rounded = round(val)
            if abs(val - rounded) > 1e-7 or rounded not in (0, 1):
                valid = False
                break
            pivot_vals.append(rounded)

        if not valid:
            continue

        candidate = [0] * N

        for bit, column in enumerate(free_columns):
            candidate[column] = free_vals[bit]

        for row_index, pivot_column in enumerate(pivots):
            candidate[pivot_column] = pivot_vals[row_index]

        candidate = tuple(candidate)

        if pyramid(candidate) == list(coefficients):
            solutions.append(candidate)

    return solutions


def demonstrate_symmetry(rng):
    word = tuple(rng.randint(0, 1) for _ in range(N))

    transformed = symmetry_operator(word, 0, N)
    original_key = symmetry_key(word)
    transformed_key = symmetry_key(transformed)

    print("Symmetry demonstration")
    print("Original word:", word)
    print("Transformed word:", transformed)
    print("Canonical keys equal:", original_key == transformed_key)
    print("Orbit sample size:", len(symmetry_orbit(word, limit=256)))


def experiment(matrix, rng, anchor_count=48, repetitions=5):
    print("\nReconstruction experiments")

    for iteration in range(repetitions):
        word = tuple(rng.randint(0, 1) for _ in range(N))
        coefficients = pyramid(word)

        selected = sorted(
            rng.sample(range(N), anchor_count)
        )

        anchors = {
            index: coefficients[index]
            for index in selected
        }

        candidates = reconstruct(
            anchors,
            coefficients,
            matrix
        )

        keys = {
            symmetry_key(candidate)
            for candidate in candidates
        }

        print(
            {
                "iteration": iteration + 1,
                "anchors": anchor_count,
                "candidate_count": len(candidates),
                "symmetry_class_count": len(keys),
                "original_recovered": word in candidates
            }
        )


def polynomial_add(left, right):
    result = dict(left)

    for monomial, coefficient in right.items():
        result[monomial] = (
            result.get(monomial, Fraction(0)) + coefficient
        )

        if result[monomial] == 0:
            del result[monomial]

    return result


def polynomial_negate(poly):
    return {
        monomial: -coefficient
        for monomial, coefficient in poly.items()
    }


def polynomial_multiply(left, right):
    result = {}

    for left_monomial, left_coefficient in left.items():
        for right_monomial, right_coefficient in right.items():
            monomial = tuple(
                left_monomial[index] + right_monomial[index]
                for index in range(2)
            )

            result[monomial] = (
                result.get(monomial, Fraction(0))
                + left_coefficient * right_coefficient
            )

    return {
        monomial: coefficient
        for monomial, coefficient in result.items()
        if coefficient != 0
    }


def polynomial_power(poly, exponent):
    if exponent < 0:
        raise ValueError("Negative exponents are not supported.")

    result = {(0, 0): Fraction(1)}
    base = poly
    power = exponent

    while power:
        if power & 1:
            result = polynomial_multiply(result, base)

        power >>= 1

        if power:
            base = polynomial_multiply(base, base)

    return result


def polynomial_from_expression(expression):
    operation = expression[0]

    if operation == "a":
        return {(1, 0): Fraction(1)}

    if operation == "b":
        return {(0, 1): Fraction(1)}

    if operation == "add":
        return polynomial_add(
            polynomial_from_expression(expression[1]),
            polynomial_from_expression(expression[2])
        )

    if operation == "sub":
        return polynomial_add(
            polynomial_from_expression(expression[1]),
            polynomial_negate(
                polynomial_from_expression(expression[2])
            )
        )

    if operation == "mul":
        return polynomial_multiply(
            polynomial_from_expression(expression[1]),
            polynomial_from_expression(expression[2])
        )

    if operation == "pow":
        return polynomial_power(
            polynomial_from_expression(expression[1]),
            expression[2]
        )

    raise ValueError(operation)


def minimum_multiplications(exponent):
    if exponent <= 1:
        return 0

    chains = [(1,)]

    for depth in range(1, exponent + 1):
        next_chains = []

        for chain in chains:
            largest = chain[-1]

            for left, right in combinations_with_replacement(chain, 2):
                value = left + right

                if value > exponent or value <= largest:
                    continue

                candidate = chain + (value,)

                if value == exponent:
                    return depth

                next_chains.append(candidate)

        chains = next_chains

    raise ValueError("Unable to construct an addition chain.")


def expression_cost(expression):
    weights = {
        "mul": 1.0,
        "add": 1.0,
        "sub": 1.0
    }

    visited = set()
    counts = {
        "mul": 0,
        "add": 0,
        "sub": 0
    }

    def visit(node):
        if node in visited:
            return

        visited.add(node)
        operation = node[0]

        if operation in ("a", "b"):
            return

        if operation == "pow":
            visit(node[1])
            counts["mul"] += minimum_multiplications(node[2])
            return

        visit(node[1])
        visit(node[2])
        counts[operation] += 1

    visit(expression)

    weighted_cost = sum(
        counts[operation] * weights[operation]
        for operation in counts
    )

    return {
        "multiplications": counts["mul"],
        "additions": counts["add"],
        "subtractions": counts["sub"],
        "estimated_cost": weighted_cost
    }


def algebraic_candidates():
    a = ("a",)
    b = ("b",)

    sum_ab = ("add", a, b)
    difference_ab = ("sub", a, b)

    product_ab = ("mul", sum_ab, difference_ab)
    difference_squares = (
        "sub",
        ("pow", a, 2),
        ("pow", b, 2)
    )

    a4b2 = ("mul", ("pow", a, 4), ("pow", b, 2))
    three_a4b2 = ("add", a4b2, ("add", a4b2, a4b2))
    a2b4 = ("mul", ("pow", a, 2), ("pow", b, 4))
    three_a2b4 = ("add", a2b4, ("add", a2b4, a2b4))

    expanded = (
        "sub",
        (
            "add",
            ("sub", ("pow", a, 6), three_a4b2),
            three_a2b4
        ),
        ("pow", b, 6)
    )

    return {
        "factored_product": (
            "mul",
            ("pow", sum_ab, 3),
            ("pow", difference_ab, 3)
        ),
        "cubed_product": (
            "pow",
            product_ab,
            3
        ),
        "difference_of_squares": (
            "pow",
            difference_squares,
            3
        ),
        "expanded_form": expanded
    }


def select_expression():
    candidates = algebraic_candidates()
    reference = polynomial_from_expression(
        candidates["factored_product"]
    )

    valid_candidates = []

    for name, expression in candidates.items():
        polynomial = polynomial_from_expression(expression)

        if polynomial != reference:
            continue

        valid_candidates.append(
            (expression_cost(expression), name, expression)
        )

    valid_candidates.sort(
        key=lambda item: (
            item[0]["estimated_cost"],
            item[0]["multiplications"],
            item[0]["additions"] + item[0]["subtractions"]
        )
    )

    print("\nEquivalent expression selection")

    for cost, name, _ in valid_candidates:
        print(name, cost)

    if valid_candidates:
        best_cost, best_name, best_expression = valid_candidates[0]

        print("\nSelected expression:", best_name)
        print("Selected cost:", best_cost)

        return best_expression

    raise RuntimeError("No equivalent candidate was found.")


def main():
    rng = Random(RANDOM_SEED)

    matrix = transformation_matrix()

    print("Configuration")
    print("N:", N)
    print("Level sizes:", LEVEL_SIZES)
    print("Level ends:", LEVEL_ENDS)
    print("Transformation matrix dimensions:", len(matrix), "x", len(matrix[0]))

    sample = tuple(rng.randint(0, 1) for _ in range(N))
    transformed = pyramid(sample)

    print("Input length:", len(sample))
    print("Output length:", len(transformed))
    print("Transformation verified:", len(transformed) == N)

    demonstrate_symmetry(rng)
    experiment(matrix, rng)
    select_expression()


if __name__ == "__main__":
    main()
