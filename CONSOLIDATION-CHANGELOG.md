# robotics — Consolidation Changelog

**Date:** 2026-06-30
**Phase:** Phase 2 Part 7 meta-repo (Model A — subtree fold)
**Tracking:** `governance/portfolio-ops/PHASE2-PART7-PILLAR-FOLD-PLAN.md`

---

## Summary

| Metric | Before | After |
|--------|-------:|------:|
| GitHub repos (this pillar) | 3 | **1** |

## Sub-project mapping

| Archived GitHub repo | Branch | Subfolder | Import commit |
|----------------------|--------|-----------|---------------|
| `vgandhi1/RL-Pendulum` | `master` | `RL-Pendulum/` | `9ac0d68` |
| `vgandhi1/semantic-SLAM-Rover` | `main` | `Semantic-SLAM-Rover/` | `1701588` |
| `vgandhi1/vla-bench` | `main` | `VLA-bench/` | `76f911e` |

**Method:** `git subtree add` per sub (history preserved). Nested `.git` dirs removed; single `origin`.
**Backup:** `/tmp/robotics-gitdirs-20260630.tgz`
**Note:** `Semantic-SLAM-Rover` had a 105M local `.git`, but the bloat was unreachable from `main` — subtree import is 1.5M, no de-bloat needed.

## Siblings archived

All three standalone repos archived (2026-06-30) with README redirect banners → this meta's subfolder.

## Pending (portfolio polish, not blocking)

- [ ] Pages workflow — subs have no `presentation.html`; add per-sub `index.html` landing then combined `_site/<sub>/` publish.
