"""The graph oracle (#2951) as a verifiers v1 taskset for prime-rl: one task per question of a questions.py JSONL, the
reward the verifier's (serve_scores.py over HTTP: minus the curve area, an answer that cannot run scored as the empty
answer). With task.revise an episode has two turns: the answer, the verifier's report on it and the request to revise
as the next user turn (serve_scores' "feedback"), and the revised answer, which the reward scores; the first answer's
reward is recorded as the metric first_reward.

  env.taskset.id = "graph-oracle", env.taskset.questions = "<questions.py JSONL>",
  env.taskset.task.seed = 0 (training) | 1000003 (held-out evaluation, train.py's --eval-seed), env.taskset.task.revise
"""

import json
from pathlib import Path

import httpx

import verifiers.v1 as vf


class GraphOracleData(vf.TaskData):
    task: str
    """The question's text id (questions.py's info.task): serve_scores reads TEXTS/<task>.json."""


class GraphOracleTaskConfig(vf.TaskConfig):
    scorer_url: str = "http://127.0.0.1:8765/score"
    seed: int = 0
    """The verifier's experiment seed: the changed prompts its curve is measured on."""
    revise: bool = False
    """Two turns: the answer, serve_scores' feedback on it, the revised answer (scored)."""
    timeout: float = 3600.0
    """Seconds one scoring request may take (a text's first request computes its baselines)."""


async def score(config: GraphOracleTaskConfig, task: str, reply: str) -> dict:
    async with httpx.AsyncClient(timeout=config.timeout) as client:
        r = await client.post(config.scorer_url, json={"task": task, "answers": [reply], "seed": config.seed})
    body = r.json()
    if r.status_code != 200:
        raise RuntimeError(f"serve_scores: {body.get('error')}")
    return body["scores"][0]


class GraphOracleTask(vf.Task[GraphOracleData, vf.State, GraphOracleTaskConfig]):
    @vf.reward(weight=1.0)
    async def verifier(self, trace: vf.Trace) -> float:
        replies = trace.assistant_messages
        s = await score(self.config, self.data.task, (replies[-1].content or "") if replies else "")
        trace.record_metric("ran", float(s["area"] is not None))
        if "first_reward" in trace.info:
            trace.record_metric("first_reward", trace.info["first_reward"])
        return s["reward"]


class GraphOracleEnv(vf.SingleAgentEnv):
    """One answer, or with task.revise the answer, the verifier's feedback as the next user turn, and the revision."""

    async def run(self, task, agents):
        async with agents.agent.interaction(task) as interaction:
            first = await interaction.turn()
            if not task.config.revise or first.terminated:
                return
            s = await score(task.config, task.data.task, first.last_reply)
            interaction.trace.info["first_reward"] = s["reward"]
            await interaction.turn(s["feedback"])


class GraphOracleConfig(vf.TasksetConfig):
    questions: str = ""
    """questions.py's JSONL: {"prompt": [user message], "info": {"task", "split"}} per question."""
    task: GraphOracleTaskConfig = GraphOracleTaskConfig()


class GraphOracleTaskset(vf.Taskset[GraphOracleTask, GraphOracleConfig]):
    def load(self) -> list[GraphOracleTask]:
        rows = [json.loads(line) for line in Path(self.config.questions).read_text().splitlines() if line.strip()]
        return [
            GraphOracleTask(GraphOracleData(id=r["info"]["task"], prompt=r["prompt"][0]["content"], task=r["info"]["task"]), self.config.task)
            for r in rows
        ]
