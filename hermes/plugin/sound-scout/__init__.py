"""Agent half of the Sound Scout Hermes package: starts the delivery loop of final job notes."""

from . import delivery


def register(ctx):
    delivery.start()
    return None
