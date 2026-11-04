"""Newton–Schulz coefficient schedules."""

from operator import index

# INT8 safety coefficients, in application order f1 -> f5.
_INT8_SAFE_NS_COEFFS = (
    (5.8918421975177555, -16.720493089530553, 11.910583457755012),
    (2.494310935454493, -3.412126511826868, 1.2164669452497787),
    (3.5701187094532, -4.7395286854633145, 1.6544471296026368),
    (2.560082131384165, -3.449401536069734, 1.2608927815298547),
    (3.8985650931142084, -7.180169545826712, 4.424064562981702),
)
_DEFAULT_NS_COEFFS = list(_INT8_SAFE_NS_COEFFS)

_POLAR_EXPRESS_BASE_COEFFS = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
)
# Apply the supplied input safety factor exactly once to the first seven
# polynomials. The eighth polynomial, also used for extra steps, stays intact.
_POLAR_EXPRESS_NS_COEFFS = tuple(
    (a / 1.01, b / 1.01**3, c / 1.01**5)
    for a, b, c in _POLAR_EXPRESS_BASE_COEFFS[:-1]
) + _POLAR_EXPRESS_BASE_COEFFS[-1:]


def recommend_coefficients(iters=5, *, method="int8_safe"):
    """Return an INT8 safety or Polar Express coefficient schedule.

    ``int8_safe`` requires at least five steps and repeats stage 3;
    seven steps apply f1, f2, f3, f3, f3, f4, f5. Polar Express takes a prefix
    of the eight supplied stages, then repeats its unscaled final polynomial.
    Additional iterations are an explicit extension, not a training guarantee.
    """
    if isinstance(iters, bool):
        raise TypeError("iters must be an integer")
    iters = index(iters)
    if method == "polar_express":
        if iters < 1:
            raise ValueError("Polar Express requires at least one iteration")
        return (list(_POLAR_EXPRESS_NS_COEFFS[:iters])
                + [_POLAR_EXPRESS_NS_COEFFS[-1]] * max(0, iters - len(_POLAR_EXPRESS_NS_COEFFS)))
    if method != "int8_safe":
        raise ValueError("method must be 'int8_safe' or 'polar_express'")
    if iters < 5:
        raise ValueError("At least five iterations are required for this schedule")
    return list(_INT8_SAFE_NS_COEFFS[:2] + (_INT8_SAFE_NS_COEFFS[2],) * (iters - 4)
                + _INT8_SAFE_NS_COEFFS[3:])
