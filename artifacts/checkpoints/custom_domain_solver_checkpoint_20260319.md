# Custom Domain Solver Checkpoint

Date: 2026-03-19

## Milestone
Generalized custom-domain solver proof completed.

## Verified facts
- Backend launches real optimizer as subprocess.
- Backend passes run-specific domain.json via LPBF_DOMAIN_JSON.
- Optimizer consumes custom geometric domain contracts.
- domain_custom supports:
  - supports
  - loads
  - non_design_regions
- non_design_regions are honored by optimizer as frozen/non-design areas.
- LPBF-aware losses remain active.

## Verified successful runs

### Custom cantilever
- run_id: run_20260319_193215_9807b3c9
- status: DONE
- health: OK

### Custom bridge
- run_id: run_20260319_225744_b2c30580
- status: DONE
- health: OK

## Proof of different outputs

### Final checkpoint tensor hashes
- cantilever theta_phys_iter_000010.pt:
  fc535cce08f71b5800ca3ab1ea81ac317ce2ab77ee8af511855a34c111b005dd
- bridge theta_phys_iter_000010.pt:
  8925dd5a08d0492a1f96163cf030fdc8f425df379e1499d9c65fcadaafedfd21

### Preview STL hashes
- cantilever preview_iter_000010_lvl_0.525.stl:
  e3fe1cfbefc2fb91d99898bde51150fcdf7c8d4d89df9e097d44650b603d509d
- bridge preview_iter_000010_lvl_0.525.stl:
  c8171b7255c7cbee02968dc51293637e6f9f27e7969891c266def5df12269a96

### Preview watertight STL hashes
- cantilever preview_watertight_iter_000010_lvl_0.525_pitch_0.250.stl:
  a32d001685e35c92c322c4f69af5086b267ff38728207f89ed7febc536104a31
- bridge preview_watertight_iter_000010_lvl_0.525_pitch_0.250.stl:
  1e0b5e79306e8823c9455fc27d3908b99da00e2544925e8f57226bbe4484f4d7

## Current UI/product scope
Supported STL import scope for v1:
- simple watertight single-body envelope STL
- coarse allowable design space only
- must fit voxel/domain limits

Not supported in v1:
- detailed production CAD
- thin-feature meshes
- lattices
- scans
- fine-detail geometries
- arbitrary exact-shape optimization

## Next step
Wire domain_custom into the UI with clear STL limitations.
