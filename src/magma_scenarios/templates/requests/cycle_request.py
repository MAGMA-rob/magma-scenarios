from typing import Dict, Any, List
import random

from magma_core.base.stage import BaseTaskStage
from magma_core.base.user_request import BaseRequest
from magma_core.base.state import TaskState
from magma_core.base.data_structures import UserInstruction

from magma_scenarios.templates.stages import MissingInformationStage, ForbiddenElemStage
from magma_scenarios.templates.constraints import ObjectAssignmentConstraint

