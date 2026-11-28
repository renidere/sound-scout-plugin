"""Agent half of the Sound Scout Hermes package: the chat binding of MCP jobs, the seen mark of
an attempt read in its chat, the sent recording in the chat and the delivery loop of final job
notes.

`register(ctx)` registers two `post_tool_call` hooks, `binding` (decision 0025) and `seen`
(decision 0030; Hermes v0.23.3 `caafe5389a`, `hermes_cli/plugins_context.py` `register_hook`),
the `pre_llm_call` and `transform_llm_output` hooks of `recording_media`, which put the player of
the turn's recording under the reply (decision 0039; without `hold_stream`, so the stream is never
held), and starts the delivery loop. A reload of the plugins clears the hooks before the new
registration, and a second `jobs bind` of one job is refused and a second `jobs seen` changes
nothing anyway, so there is no guard against a second registration.
"""

from . import binding, delivery, recording_media, seen


def register(ctx):
    ctx.register_hook("post_tool_call", binding.post_tool_call)
    ctx.register_hook("post_tool_call", seen.post_tool_call)
    ctx.register_hook("pre_llm_call", recording_media.pre_llm_call)
    ctx.register_hook("transform_llm_output", recording_media.transform_llm_output)
    delivery.start()
    return None
