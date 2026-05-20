# V 1.0

`rl-toybox` V 1.0 is the first stable public version of the project.

This release packages six compact reinforcement-learning examples:

- `snake`: value methods / Q-learning-style control
- `bang`: DQN-style discrete arena control
- `jump`: PPO / on-policy actor-critic
- `vroom`: SAC / continuous control
- `flip`: MCTS + self-play
- `kick`: multi-agent CTDE

The repo includes shared training, runtime, logging, configuration, rendering, and run-artifact infrastructure while keeping each game small enough to inspect end to end.

Suggested learning path:

1. Start with `snake` for the smallest discrete value-control example.
2. Move to `bang` for richer DQN-style control.
3. Read `jump` for PPO and on-policy actor-critic.
4. Try `vroom` for continuous SAC control.
5. Explore `flip` for planning and self-play.
6. Finish with `kick` for shared-policy multi-agent CTDE.

V 1.0 also includes a GitHub Actions smoke test that installs the package and boots/resets/steps every environment under a virtual display.

This is intentionally a compact toybox, not a full RL zoo.
