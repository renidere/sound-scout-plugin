"""Agent half of the Sound Scout Hermes package: the chat binding of MCP jobs and the delivery
loop of final job notes.

`register(ctx)` registers the `post_tool_call` hook of `binding` (decision 0025; Hermes v0.23.3
`caafe5389a`, `hermes_cli/plugins_context.py` `register_hook`) and starts the delivery loop. A
reload of the plugins clears the hooks before the new registration, and a second `jobs bind` of
one job is refused anyway, so there is no guard against a second registration.
"""

from . import binding, delivery


def register(ctx):
    ctx.register_hook("post_tool_call", binding.post_tool_call)
    delivery.start()
    return None
