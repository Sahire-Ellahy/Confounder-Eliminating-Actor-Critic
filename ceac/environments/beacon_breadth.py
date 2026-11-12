"""Physical false-beacon chamber; the historical shoreline kernel is unchanged."""
from dataclasses import dataclass
import numpy as np
from . import frozenlake as shore

# Nested connected footprints, all outside the original open corridors.
FIELD_ORDER = ((3, 2), (3, 1), (2, 2), (3, 0), (2, 1), (1, 2), (2, 0), (1, 1))
GATEWAY = shore.SHALLOW_BUOY
ENTRY = FIELD_ORDER[0]
MOVES = {shore.LEFT:(0,-1), shore.DOWN:(1,0), shore.RIGHT:(0,1), shore.UP:(-1,0)}


@dataclass(frozen=True)
class BeaconBreadthConfig(shore.FrozenShoreConfig):
    beacon_cells: int = 0

    def __post_init__(self):
        super().__post_init__()
        if type(self.beacon_cells) is not int or not 0 <= self.beacon_cells <= len(FIELD_ORDER):
            raise ValueError("beacon_cells must be an integer from zero to eight")
        if self.nuisance_probability is not None:
            raise ValueError("The breadth experiment does not include probabilistic dwell")
        if (self.rows,self.cols)!=(7,9):
            raise ValueError("Breadth uses the registered 7 by 9 shoreline")


class BeaconBreadthEnv(shore.ShiftingShoreFrozenLakeEnv):
    def __init__(self, config=None):
        cfg=config or BeaconBreadthConfig()
        self.field_cells=frozenset(FIELD_ORDER[:cfg.beacon_cells])
        self.open_cells=shore.OPEN_CELLS | self.field_cells
        super().__init__(cfg)

    def valid_action_mask(self, observation=None):
        state=self._state() if observation is None else self.decode(observation)
        here=(state.row,state.col)
        mask=np.zeros(4,dtype=np.int8)
        for action,(dr,dc) in MOVES.items():
            there=(state.row+dr,state.col+dc)
            original=here in shore.OPEN_CELLS and there in shore.OPEN_CELLS
            internal=here in self.field_cells and there in self.field_cells
            gate=bool(self.field_cells) and {here,there}=={GATEWAY,ENTRY}
            mask[action]=original or internal or gate
        return mask

    def step(self, action):
        # The base simulator consults this subclass's mask before moving.
        # Thus adjacent old/new cells have walls except at the registered gate.
        obs,reward,term,trunc,info=super().step(action)
        in_field=(self.row,self.col) in self.field_cells
        info["beacon_field_step"]=in_field
        if in_field and info["diagnostic_event"]=="none":
            info["diagnostic_event"]="beacon_field"
        return obs,reward,term,trunc,info

    def render(self):
        lines=[]
        for row in range(self.config.rows):
            tokens=[]
            for col in range(self.config.cols):
                p=(row,col)
                tokens.append("A" if p==(self.row,self.col) else "B" if p in self.field_cells
                              else "." if p in shore.OPEN_CELLS else "#")
            lines.append("".join(tokens))
        return "\n".join(lines)+"\nB = beacon chamber; its only entrance is the shallow buoy."
