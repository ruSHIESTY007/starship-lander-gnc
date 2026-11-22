# starship-lander-gnc

Guidance, navigation & control flight software for a **generic belly-flop reusable vehicle**:
aerodynamic glide steered by flaps → engine-lit flip → landing burn → vertical terminal
descent onto a precision pad, with online wind estimation and an engine-out abort that
diverts to a backup site.

> The vehicle is notional (round-number mass/geometry/engine values). It is "Starship-style" in
> its flight profile only; it is not a model of any real vehicle.

* **Flight software** — C99, single precision, no dynamic memory (`fsw/`)
* **Truth simulation** — Python 6-DOF, standard atmosphere, Mach/AoA aero, wind + Dryden gusts (`sim/`)
* **Software-in-the-loop** — the C FSW is built as `libfsw.so` and driven from Python via `ctypes`
* **Analysis** — convex-optimal (G-FOLD) benchmark, learned guidance, Monte Carlo (`analysis/`)

_Work in progress — see commit history for milestones._

## Build
```bash
make venv        # python deps into .venv
make fsw         # builds fsw/build/libfsw.so
make test        # C unit tests + pytest
```
