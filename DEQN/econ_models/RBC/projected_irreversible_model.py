"""Saving-rate RBC with an explicit investment floor."""

from DEQN.econ_models.RBC.model import _SavingRateModel


class ProjectedIrreversibleModel(_SavingRateModel):
    """Irreversible RBC for projected or penalty-based policy optimization."""

    def __init__(self, *args, i_min_frac, **kwargs):
        if not 0 < float(i_min_frac) < 1:
            raise ValueError(
                "ProjectedIrreversibleModel requires 0 < i_min_frac < 1"
            )
        super().__init__(*args, i_min_frac=i_min_frac, **kwargs)
