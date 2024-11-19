# Copied from https://github.com/RussTedrake/underactuated/blob/7faf2721c248e889464fe024a65982a43dd78aff/underactuated/multibody.py
# See https://github.com/RussTedrake/underactuated/blob/7faf2721c248e889464fe024a65982a43dd78aff/examples/double_pendulum/dynamics.ipynb
# for usage info (last two cells)
# TODO(ndobrad):update dockerfile to install the underactuated project
# (https://github.com/RussTedrake/underactuated/blob/7faf2721c248e889464fe024a65982a43dd78aff/scripts/setup/jupyter_setup.py#L52)
from pydrake.autodiffutils import AutoDiffXd
from pydrake.multibody.tree import MultibodyForces_
from pydrake.multibody.plant import MultibodyPlant_
from pydrake.symbolic import Expression
import numpy as np


def ManipulatorDynamics(plant, q, v=None):
    context = plant.CreateDefaultContext()
    plant.SetPositions(context, q)
    if v is not None:
        plant.SetVelocities(context, v)
    M = plant.CalcMassMatrixViaInverseDynamics(context)
    Cv = plant.CalcBiasTerm(context)
    tauG = plant.CalcGravityGeneralizedForces(context)
    B = plant.MakeActuationMatrix()
    forces = MultibodyForces_(plant)
    plant.CalcForceElementsContribution(context, forces)
    tauExt = forces.generalized_forces()

    return (M, Cv, tauG, B, tauExt)