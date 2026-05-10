# Karma agents

Agents are first-class Karma control producers. An agent receives an `obs` dict
and returns an action dict.

Common agents:

- `karma.agents.teleoperation.yam_leader_agent:YamLeaderAgent`
- `karma.agents.teleoperation.gello_leader_agent:GelloLeaderAgent`
- `karma.agents.teleoperation.passive_gello_leader_agent:PassiveGelloLeaderAgent`
- `karma.agents.policy_learning.async_remote_act_agent:AsyncRemoteACTAgent`
- `karma.agents.policy_learning.async_pi0_agent:AsyncDiffusionAgent`
- `karma.agents.policy_learning.async_molmoact2_agent:AsyncMolmoAct2Agent`

Wire agents with `AgentNode` in YAML and use the standard bus names
`leader_left`, `leader_right`, `left`, and `right`.
