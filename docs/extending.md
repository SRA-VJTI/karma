# Extending Karma

Karma extension points are plain Python classes referenced from YAML with
`module.path:ClassName` strings.

## Add an agent

Implement `act(obs) -> dict` and optionally `reset()` / `close()`.

```python
# karma/agents/my_agent.py
from karma.agents.agent import Agent


class MyAgent(Agent):
    def act(self, obs):
        return {"left": {"pos": obs["left"]["joint_pos"]}}
```

Wire it into a session:

```yaml
- type: AgentNode
  name: my_policy
  agent_class: karma.agents.my_agent:MyAgent
  state_topics:
    left: left/joint_state
    right: right/joint_state
  loop_mode: fixed_rate
  poll_freq: 30
```

`AgentNode` publishes `joint_pos` for single-arm agents or `{arm}_pos` for
multi-arm dictionaries.

## Add a camera driver

Implement the `CameraDriver` interface from
`karma.sensors.cameras.camera`. Then register it in
`karma.runtime.environment.camera_node` or pass a fully qualified class path.

## Add a session config

Prefer the standard names:

```yaml
session:
  save_root: recordings
  record_topic: leader_left/record

nodes:
  - type: AgentNode
    name: leader_left
    agent_class: karma.agents.teleoperation.gello_leader_agent:GelloLeaderAgent

  - type: RobotNode
    name: left
    robot_config: robot_configs/left.yaml
    cmd_topic: leader_left/joint_pos
```
