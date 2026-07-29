"""Agent under test.

Two agents are evaluated: a frontier model and an OSS model. They share this entire
package — the same loop, prompts, tool protocol, and tools — so that the only variable
between them is the model behind `agent.models`. See PROJECT.md.
"""
