# robotics — Robotics Algorithms Pillar

Three robotics projects across the train → deploy → perceive lifecycle:

| Sub-project | Focus | Stack |
|-------------|-------|-------|
| [`VLA-bench`](VLA-bench/) | Vision-Language-Action model training benchmark | WebDataset, FSDP |
| [`RL-Pendulum`](RL-Pendulum/) | Sim-to-real reinforcement learning, edge deploy | TFLite |
| [`Semantic-SLAM-Rover`](Semantic-SLAM-Rover/) | Semantic SLAM navigation | ROS2 |

**Disambiguation:** this pillar is robotics *algorithms* (training, control, perception).
For robotics *operations data* (HITL failure recovery, teleop data quality) see
[`robot-data-flywheel`](https://github.com/vgandhi1/robot-data-flywheel).

_Consolidated 2026-06-30 from standalone repos `RL-Pendulum`, `semantic-SLAM-Rover`, `vla-bench` (history preserved via git subtree; originals archived)._
