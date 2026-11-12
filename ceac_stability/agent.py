"""Reuse the frozen CEAC selector and replay; replace only its actor optimizer."""
from ceac_pseudocode.agent import CEAC,Config
from .actor import StabilityActor

class StabilityCEAC(CEAC):
    def __init__(self,model,rng,candidate,schedule):
        super().__init__(model,rng,Config())
        # Base initialization creates statistical debts, but executes no updates.
        if self.actor.updates or self.episodes:raise AssertionError('Actor replacement must precede observations')
        self.actor=StabilityActor(model,candidate,schedule)

    def diagnostics(self):
        result=super().diagnostics()
        result['algorithm']='CEAC_original_reference' if self.actor.reference else 'CEAC_optimizer_variant'
        result['optimizer']=self.actor.candidate
        result['geometry_diagnostics']=self.actor.diagnostics()
        return result
