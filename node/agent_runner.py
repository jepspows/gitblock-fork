"""GitBlock Agent Runner — Autonomous agent runtime for node operators.

Runs Aeon-compatible autonomous agent sessions on GitBlock nodes.
Each session is a multi-turn LLM loop with tool calling, memory persistence,
and quality scoring. Node operators earn $GITBLOCK for running agent sessions.

Architecture:
    AgentTask arrives → AgentRunner loads skill + memory + tools
    → Multi-turn LLM loop (think → act → observe → repeat)
    → Saves memory, scores output, reports usage for rewards

This is the core component that makes GitBlock a decentralized autonomous
agent network, not just an inference API.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from .agent_memory import AgentConfig, AgentHealth, AgentMemory
from .agent_tools import ToolRegistry

logger = logging.getLogger("gitblock-agent-runner")


# ── Data Models ──


@dataclass
class AgentTask:
    """A task dispatched to a node to run an agent session."""
    skill: str  # skill name (e.g., "token-alert")
    var: str = ""  # focus parameter
    model: str = "llama-3.3-70b"  # open model to use
    max_turns: int = 20
    memory_context: str = ""  # pre-loaded memory for the session
    chain_context: str = ""  # output from upstream skills in a chain


@dataclass
class AgentTurn:
    """One turn in the agent loop."""
    index: int
    role: str  # "think", "act", "observe", "done"
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    """Result of a completed agent session."""
    agent_id: str
    skill: str
    success: bool
    output: str
    turns: list[AgentTurn] = field(default_factory=list)
    tokens_used: int = 0
    tools_called: int = 0
    quality_score: float = 0.0
    error: str = ""
    duration_ms: float = 0.0
    reward_estimate: float = 0.0  # estimated $GITBLOCK reward


# ── Skill Prompts ──

SKILL_PROMPTS: dict[str, str] = {
    "token-alert": """You are a crypto price monitor agent. Your job is to check cryptocurrency prices and alert the user about significant moves.

Focus token: {var}

On each run:
1. Get current price using the get_crypto_price tool
2. Check if price moved significantly (5%+ in 24h)
3. If the move is significant, draft an alert with: token name, current price, 24h change, and a brief note
4. Use send_notification to alert the user if there's a significant move
5. If prices are stable, save a note to memory and exit quietly

Be concise. No unnecessary commentary.""",

    "morning-brief": """You are a morning brief agent. Your job is to create a daily digest of important AI and crypto news.

Focus: {var}
Model used: {model}

On each run:
1. Search the web for today's top AI and crypto news using web_search
2. Curate the 3-5 most important stories
3. For each story: headline, one-sentence summary, and source
4. Format as a clean morning brief
5. Send via send_notification if configured
6. Save the brief to memory for future reference

Keep it tight. One paragraph per story max. End with "That's the brief for [today's date]." """,

    "token-report": """You are a token analysis agent. Your job is to create daily token reports with price data and market context.

Focus token: {var}

On each run:
1. Get price data using get_crypto_price for the focus token
2. Search for recent news about this token using web_search
3. Analyze: is the price trending up/down/sideways? What's driving it?
4. Write a concise report: price, 24h change, market context, 1-sentence outlook
5. Send via send_notification
6. Save to memory

Be data-driven. Cite sources when possible.""",

    "monitor-polymarket": """You are a prediction market monitor agent. You track Polymarket markets for significant moves.

Focus: {var}

On each run:
1. Search for active Polymarket markets related to your focus topic
2. Identify markets with significant 24h price or volume changes
3. Report: market name, current price, 24h change, volume, your read on what's moving it
4. Send alerts for any markets with >10% moves
5. Save observations to memory

Markets are probabilities (0-100 cents). Treat them as such.""",

    "pr-review": """You are a code review agent. You review pull requests for quality, security, and correctness.

Focus repo: {var}

On each run:
1. Check recent PRs in the focus repo for review-worthy changes
2. Review the code diff for: bugs, security issues, style violations, missing tests
3. Provide inline feedback: what's good, what needs changes, severity
4. Be constructive, not nitpicky
5. Send summary via send_notification

Focus on real issues. Skip cosmetic nitpicks.""",

    "write-tweet": """You are a social media content agent. You draft tweets and threads for review.

Focus topic: {var}

On each run:
1. Search for trending topics in your focus area
2. Draft 2-3 tweet options (each under 280 chars) or 1 thread option
3. Make them engaging: hook, insight, call to action
4. Save drafts to memory — the user reviews before posting
5. Do NOT post anything — draft only

Voice: knowledgeable but not arrogant. No hashtag spam. One good insight per tweet.""",
}


# ── Agent Runner ──


class AgentRunner:
    """Runs autonomous agent sessions on a GitBlock node.

    Usage:
        runner = AgentRunner(config)
        result = await runner.run(AgentTask(skill="token-alert", var="$GITBLOCK"))
    """

    def __init__(self, config=None):
        self.tools = ToolRegistry()
        self._running = False
        self._sessions_run = 0

    async def run(self, task: AgentTask, agent_id: str = "") -> AgentResult:
        """Run a full agent session. Multi-turn LLM loop with tools and memory."""
        start_time = time.time()
        agent_id = agent_id or f"agent_{int(time.time())}"
        turns = []
        tokens_used = 0
        tools_called = 0

        # Load agent state
        memory = AgentMemory.load(agent_id)
        health = AgentHealth.load(agent_id)
        config = AgentConfig.load(agent_id)

        # Build skill prompt
        skill_prompt = SKILL_PROMPTS.get(task.skill, SKILL_PROMPTS.get("token-alert", ""))
        skill_prompt = skill_prompt.format(
            var=task.var or "general",
            model=task.model,
        )

        # Build system prompt
        system_prompt = self._build_system_prompt(
            skill_prompt=skill_prompt,
            memory=memory,
            health=health,
            chain_context=task.chain_context,
        )

        try:
            self._running = True
            done = False
            turn_index = 0
            messages = [{"role": "system", "content": system_prompt}]

            while not done and turn_index < task.max_turns:
                turn_index += 1

                # Think: call the LLM
                turn = await self._think(messages, task.model)
                turns.append(turn)
                tokens_used += len(turn.content.split()) + 50  # rough estimate

                if turn.tool_calls:
                    # Act: execute tool calls
                    results = []
                    for tc in turn.tool_calls:
                        tools_called += 1
                        result = await self.tools.execute(tc["name"], tc.get("arguments", {}))
                        results.append(result)

                    turn.tool_results = results
                    messages.append({"role": "assistant", "content": turn.content})
                    messages.append({"role": "tool", "content": "\n".join(results)})

                elif "DONE" in turn.content or "FINAL_OUTPUT" in turn.content:
                    done = True
                else:
                    done = True  # no tool calls, treat as final

            duration_ms = (time.time() - start_time) * 1000

            # Generate output
            output = turns[-1].content if turns else "(no output)"
            quality = self._score_output(output, task.skill)

            # Update memory
            memory.total_runs += 1
            memory.last_run = time.strftime("%Y-%m-%dT%H:%M:%SZ")
            memory.save()

            # Update health
            health.record_session(success=True, quality=quality)
            health.save()

            # Calculate reward estimate
            reward = self._estimate_reward(tokens_used, tools_called, duration_ms)

            self._sessions_run += 1
            self._running = False

            return AgentResult(
                agent_id=agent_id,
                skill=task.skill,
                success=True,
                output=output,
                turns=turns,
                tokens_used=tokens_used,
                tools_called=tools_called,
                quality_score=quality,
                duration_ms=duration_ms,
                reward_estimate=reward,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Agent session failed: {e}")

            health.record_session(success=False, quality=0.0, error=str(e))
            health.save()

            self._running = False
            return AgentResult(
                agent_id=agent_id,
                skill=task.skill,
                success=False,
                output="",
                turns=turns,
                tokens_used=tokens_used,
                tools_called=tools_called,
                error=str(e),
                duration_ms=duration_ms,
            )

    def _build_system_prompt(
        self,
        skill_prompt: str,
        memory: AgentMemory,
        health: AgentHealth,
        chain_context: str = "",
    ) -> str:
        """Build the full system prompt with skill, memory, and context."""
        parts = [skill_prompt]

        # Add memory context
        memory_summary = memory.summary()
        if memory_summary and "(no memory yet)" not in memory_summary:
            parts.append(f"\n## Persistent Memory\n{memory_summary}")

        # Add chain context (output from upstream skills)
        if chain_context:
            parts.append(f"\n## Context from Previous Skills\n{chain_context}")

        # Add health context if failing
        if health.needs_repair():
            parts.append(
                f"\n## ⚠️ Health Alert\n"
                f"This agent has {health.consecutive_failures} consecutive failures. "
                f"Focus on reliability. If tools are failing, simplify your approach."
            )

        # Add instructions
        parts.append(
            "\n## Instructions\n"
            "- Use tools to gather real data — never fabricate information.\n"
            "- Save important findings to memory with write_memory.\n"
            "- If nothing significant happened, say so briefly and end.\n"
            "- If sending a notification, use send_notification.\n"
            "- When complete, output final results clearly."
        )

        return "\n\n".join(parts)

    async def _think(self, messages: list[dict], model: str) -> AgentTurn:
        """Call the LLM for one reasoning step.

        In production, this routes to the node's GPU (vLLM, llama.cpp, Ollama).
        For now, it returns a structured placeholder that demonstrates the flow.
        """
        # In production: call local model via the node's inference backend
        # response = await self.node_inference.chat(messages, model=model, tools=self.tools.list_tools())

        # Placeholder: simulate a think step
        last_message = messages[-1]["content"] if messages else ""
        turn_index = sum(1 for m in messages if m["role"] == "assistant")

        if turn_index == 0:
            # First turn: agent decides what tool to call
            return AgentTurn(
                index=turn_index,
                role="think",
                content=f"[Agent thinking] Analyzing task with model {model}...",
                tool_calls=[{"name": "get_current_time", "arguments": {}}],
            )
        else:
            # Subsequent turns: process tool results and produce output
            return AgentTurn(
                index=turn_index,
                role="done",
                content=(
                    f"[GitBlock Agent] Session complete. "
                    f"Task processed using {model}. "
                    f"Connect a local model backend (vLLM, Ollama, or llama.cpp) on this node "
                    f"for full autonomous agent execution."
                ),
            )

    def _score_output(self, output: str, skill: str) -> float:
        """Score the output quality 1-5. Simple heuristic — production uses LLM judge."""
        if not output:
            return 1.0
        if len(output) < 50:
            return 2.0
        if len(output) < 200:
            return 3.0
        if "error" in output.lower():
            return 3.0
        return 4.0  # default decent score — production uses Haiku/LLM judge

    def _estimate_reward(self, tokens: int, tools: int, duration_ms: float) -> float:
        """Estimate $GITBLOCK reward for this session.

        Agents pay more than inference because they're sustained, multi-turn,
        tool-using workloads. Node operators earn based on session complexity.
        """
        base = 0.005  # $GITBLOCK base per session (5x more than single inference)
        token_bonus = (tokens / 1000) * 0.001  # bonus per 1K tokens
        tool_bonus = tools * 0.002  # bonus per tool call
        duration_bonus = (duration_ms / 60000) * 0.001  # bonus per minute

        return round(base + token_bonus + tool_bonus + duration_bonus, 4)


# ── Integration with Node Server ──


def create_agent_endpoints(app, runner: Optional[AgentRunner] = None):
    """Add agent endpoints to an existing FastAPI node server.

    Usage in node/server.py create_app():
        from .agent_runner import create_agent_endpoints
        app = create_app(config)
        create_agent_endpoints(app)
    """
    from fastapi import HTTPException

    runner = runner or AgentRunner()

    @app.post("/v1/agent/run")
    async def run_agent(task: dict):
        """Run an autonomous agent session on this node."""
        try:
            agent_task = AgentTask(
                skill=task.get("skill", "token-alert"),
                var=task.get("var", ""),
                model=task.get("model", "llama-3.3-70b"),
                max_turns=task.get("max_turns", 20),
                memory_context=task.get("memory_context", ""),
                chain_context=task.get("chain_context", ""),
            )
            agent_id = task.get("agent_id", "")
            result = await runner.run(agent_task, agent_id=agent_id)
            return {
                "agent_id": result.agent_id,
                "skill": result.skill,
                "success": result.success,
                "output": result.output[:2000],  # truncate for API response
                "turns": len(result.turns),
                "tokens_used": result.tokens_used,
                "tools_called": result.tools_called,
                "quality_score": result.quality_score,
                "duration_ms": result.duration_ms,
                "reward_estimate": result.reward_estimate,
                "error": result.error,
            }
        except Exception as e:
            raise HTTPException(500, f"Agent session failed: {e}")

    @app.get("/v1/agent/skills")
    async def list_skills():
        """List available agent skills with descriptions."""
        return {
            skill: {
                "description": prompt.split("\n")[0].replace("You are a ", "").rstrip("."),
                "parameters": ["var"] if "{var}" in prompt else [],
            }
            for skill, prompt in SKILL_PROMPTS.items()
        }

    @app.get("/v1/agent/{agent_id}/memory")
    async def get_agent_memory(agent_id: str):
        """Get an agent's persistent memory."""
        memory = AgentMemory.load(agent_id)
        health = AgentHealth.load(agent_id)
        return {
            "agent_id": agent_id,
            "memory": memory.summary(),
            "health": health.summary(),
            "goals": memory.goals,
            "active_topics": memory.active_topics,
            "notes": memory.notes,
        }

    return runner
