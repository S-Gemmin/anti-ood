### Anticipatory OOD Safety

Run `python experiment.py` to compare reactive vs anticipatory safety controllers. The simulation models a particle moving in a 2D embedding space with a Gaussian safe cluster. Four scenarios test identical starting distances but different velocities: stationary, moving inward (toward safe), moving outward (away from safe), and moving tangent. Each runs 100 trials per controller. The key metric is boundary crossing rate. Anticipatory should show ~75% fewer crossings in the outward scenario.

Results print to console with crossing rates for each scenario and controller. The `outward` scenario improvement confirms whether anticipatory control successfully uses radial velocity to prevent boundary crossings. All parameters are in `config.py`. No output files are created; modify code to save results if needed.
