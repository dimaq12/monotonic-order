# Contributing to IdealOrder

Contributions are welcome when they preserve the project's central contract:
payloads are ordered through explicit monotonic keys and the native radix kernel
does not call arbitrary Python comparators.

## Development setup

```bash
git clone https://github.com/dimaq12/order.git
cd order
python -m pip install -e ".[test]"
python -m pytest
```

Native verification on Linux:

```bash
make test
make asan
```

## Pull requests

- add a regression or property test for every behavioral change;
- compare new order codecs with an independent trusted reference;
- state whether ordering is exact, quantized or approximate;
- include warmed medians, input distribution and environment for performance
  claims;
- keep public APIs stable and permutations stable for equal keys;
- update `THEORY.md`, benchmark reports and `CHANGELOG.md` when applicable.

Do not commit build products, generated shared libraries or benchmark claims
that cannot be reproduced from a checked-in script.
