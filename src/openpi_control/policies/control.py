"""Control intents shared by headless execution, HITL and Viser.

Callbacks only change intent. The runtime alone sends hardware commands.
"""

import threading


class PolicyControl:
    def __init__(self):
        self.lock = threading.RLock()
        self.generation = 0
        self.paused = False
        self.waiting = False
        self.human = False
        self.home = False
        self.reason = "Ready"

    def wait_for_observation(self, reason):
        """Startup-only inhibition; never clears an operator pause."""
        with self.lock:
            if not self.waiting:
                self.generation += 1
                self.waiting = True
                self.reason = reason

    def observation_ready(self):
        with self.lock:
            if self.waiting and not (self.paused or self.human or self.home):
                self.generation += 1
                self.waiting = False
                self.reason = "Fresh observations available; starting policy"

    def pause(self, reason="Policy stopped; holding"):
        with self.lock:
            self.generation += 1
            self.paused = True
            self.reason = reason

    def resume(self):
        with self.lock:
            if self.home or self.human:
                return
            self.generation += 1
            self.paused = False
            self.reason = "Reobserving"

    def home_and_stop(self):
        with self.lock:
            self.home = True
            self.pause("Homing and stopping")

    def intervention(self, active):
        with self.lock:
            self.human = active
            self.generation += 1
            self.reason = "Human in control" if active else "Reobserving"

    def permits(self, generation):
        return (
            not (self.paused or self.waiting or self.human or self.home)
            and generation == self.generation
        )
