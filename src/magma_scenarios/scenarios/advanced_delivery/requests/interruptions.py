import random
from dataclasses import dataclass
from typing import List, Literal, Optional

from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState

from ..ad_stages import (
    AtLeastCompletedObjectivesStage,
    AskDeliveryDetailsStage,
    AskDeliveryServiceCountsStage,
)
from ..environment_transitions import RECEPTION_OFFSETS, ResetAdvancedDeliveryTransition
from .common import (
    DeliveryOrder,
    deliveries_description,
    delivery_description,
    save_assignments,
)
from .orders import (
    ProcessingParameters,
    ProcessReceptionAndDeliveriesRequest,
    processing_objective_count,
)


@dataclass(frozen=True)
class QuestionInterruptionParameters:
    processing: ProcessingParameters
    question_kind: Literal["details", "service_counts"]
    selected_order_index: Optional[int]
    insertion_index: int


def _question_stage(
    parameters: QuestionInterruptionParameters,
) -> AskDeliveryDetailsStage | AskDeliveryServiceCountsStage:
    assignments = parameters.processing.assignment_dicts()
    if parameters.question_kind == "details":
        if parameters.selected_order_index is None:
            raise ValueError("A details question requires an order index.")
        return AskDeliveryDetailsStage(
            assignments[parameters.selected_order_index]
        )
    return AskDeliveryServiceCountsStage(assignments)


class _QuestionInterruption(ProcessReceptionAndDeliveriesRequest):
    question_kind: Literal["details", "service_counts"]

    def sample_parameters(
        self,
        state: TaskState,
    ) -> QuestionInterruptionParameters:
        processing = super().sample_parameters(state)
        assignments = processing.assignment_dicts()
        if self.question_kind == "details":
            selected_index = random.randrange(len(assignments))
        else:
            selected_index = None
        reception_assignment, fixed_targets = self._build_reception_assignment(
            state,
            assignments,
        )
        objective_count = processing_objective_count(
            reception_assignment,
            fixed_targets,
            len(assignments),
        )
        insertion_index = random.randint(1, objective_count - 1)
        return QuestionInterruptionParameters(
            processing,
            self.question_kind,
            selected_index,
            insertion_index,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: QuestionInterruptionParameters,
    ) -> List[BaseTaskStage]:
        stages = super().create_stages(state, parameters.processing)
        question_stage = _question_stage(parameters)
        question_stage.stage_input.linked_to_prev = True
        stages.insert(parameters.insertion_index, question_stage)
        return stages

    def apply_request(
        self,
        state: TaskState,
        parameters: QuestionInterruptionParameters,
    ) -> TaskState:
        return super().apply_request(state, parameters.processing)


class AskDeliveryDetailsInterruption(_QuestionInterruption):
    question_kind = "details"

    def __init__(
        self,
        max_deliveries: int = 2,
        max_products_per_delivery: int = 3,
        weight: float = 1.5,
    ) -> None:
        super().__init__(
            max_deliveries=max_deliveries,
            max_products_per_delivery=max_products_per_delivery,
            weight=weight,
        )


class AskDeliveryServiceCountsInterruption(_QuestionInterruption):
    question_kind = "service_counts"

    def __init__(
        self,
        max_deliveries: int = 2,
        max_products_per_delivery: int = 3,
        weight: float = 1.5,
    ) -> None:
        super().__init__(
            max_deliveries=max_deliveries,
            max_products_per_delivery=max_products_per_delivery,
            weight=weight,
        )


@dataclass(frozen=True)
class ReceptionOrDeliveriesParameters:
    processing: ProcessingParameters
    reception_first: bool

    @property
    def initial_instruction(self) -> str:
        assignments = self.processing.assignment_dicts()
        if self.reception_first:
            return "There is a client return in reception. Process it."
        return (
            f"Prepare {len(assignments)} "
            f"{'delivery' if len(assignments) == 1 else 'deliveries'}: "
            f"{deliveries_description(assignments)}."
        )

    @property
    def added_instruction(self) -> str:
        if self.reception_first:
            return (
                "There are now deliveries to prepare as well: "
                f"{deliveries_description(self.processing.assignment_dicts())}."
            )
        return (
            "A client return has now arrived in reception. Process it as well."
        )


class AddReceptionOrDeliveriesInterruption(
    ProcessReceptionAndDeliveriesRequest
):
    def __init__(
        self,
        max_deliveries: int = 2,
        max_products_per_delivery: int = 3,
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            max_deliveries=max_deliveries,
            max_products_per_delivery=max_products_per_delivery,
            weight=weight,
        )

    def sample_parameters(
        self,
        state: TaskState,
    ) -> ReceptionOrDeliveriesParameters:
        processing = super().sample_parameters(state)
        reception_first = random.choice([True, False])
        return ReceptionOrDeliveriesParameters(processing, reception_first)

    def create_stages(
        self,
        state: TaskState,
        parameters: ReceptionOrDeliveriesParameters,
    ) -> List[BaseTaskStage]:
        processing = parameters.processing
        assignments = processing.assignment_dicts()
        current_reception = list(processing.current_reception_objects)
        reception_assignment, fixed_targets = self._build_reception_assignment(
            state,
            assignments,
        )
        objective_count = processing_objective_count(
            reception_assignment,
            fixed_targets,
            len(assignments),
        )
        initial_reception_assignment, initial_fixed_targets = self._build_reception_assignment(
            state,
            [],
        )
        if parameters.reception_first:
            initial_stage = AtLeastCompletedObjectivesStage(
                physical_assignment=initial_reception_assignment,
                fixed_targets_by_object=initial_fixed_targets,
                minimum_completed_goals=1,
                instruction=parameters.initial_instruction,
                reception_product_count=len(current_reception),
                flag_answer=False,
            )
        else:
            initial_stage = AtLeastCompletedObjectivesStage(
                delivery_assignments=assignments,
                minimum_completed_goals=1,
                instruction=parameters.initial_instruction,
                flag_answer=False,
            )
        initial_stage.entry_transition = ResetAdvancedDeliveryTransition(
            reception_objects=current_reception,
            broken_objects=list(initial_fixed_targets),
            reception_offsets=processing.reception_offsets,
        )
        stages = [initial_stage]
        for minimum in range(2, objective_count + 1):
            stage = AtLeastCompletedObjectivesStage(
                physical_assignment=reception_assignment,
                fixed_targets_by_object=fixed_targets,
                delivery_assignments=assignments,
                minimum_completed_goals=minimum,
                instruction=parameters.added_instruction if minimum == 2 else "none",
                reception_product_count=len(current_reception),
                flag_answer=False,
            )
            if minimum == 2:
                stage.stage_input.linked_to_prev = True
            stages.append(stage)
        return stages

    def apply_request(
        self,
        state: TaskState,
        parameters: ReceptionOrDeliveriesParameters,
    ) -> TaskState:
        return super().apply_request(state, parameters.processing)


@dataclass(frozen=True)
class DeliveryInProgressParameters:
    processing: ProcessingParameters

    @property
    def initial_instruction(self) -> str:
        assignments = self.processing.assignment_dicts()
        return (
            "Prepare these two deliveries: "
            f"{deliveries_description(assignments[:2])}."
        )

    @property
    def added_instruction(self) -> str:
        assignment = self.processing.assignment_dicts()[2]
        return (
            "There is now another delivery to prepare as well: "
            f"{delivery_description(assignment)}."
        )


class AddDeliveryInProgressRequest(ProcessReceptionAndDeliveriesRequest):
    def __init__(
        self,
        max_products_per_delivery: int = 3,
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            max_deliveries=3,
            max_products_per_delivery=max_products_per_delivery,
            weight=weight,
        )

    def sampling_weight(self, state: TaskState) -> float:
        return self.weight if len(self._available_delivery_products(state)) >= 3 else 0

    def sample_parameters(
        self,
        state: TaskState,
    ) -> DeliveryInProgressParameters:
        available_products = self._available_delivery_products(state)
        if len(available_products) < 3:
            raise RuntimeError("Three deliveries cannot currently be created.")
        assignments = self._build_deliveries(
            state,
            available_products,
            delivery_count=3,
        )
        processing = ProcessingParameters(
            tuple(DeliveryOrder.from_assignment(item) for item in assignments),
            tuple(self._reception_objects(state)),
            (),
            (),
            tuple(
                random.sample(
                    RECEPTION_OFFSETS,
                    k=len(self._reception_objects(state)),
                )
            ),
        )
        return DeliveryInProgressParameters(processing)

    def create_stages(
        self,
        state: TaskState,
        parameters: DeliveryInProgressParameters,
    ) -> List[BaseTaskStage]:
        assignments = parameters.processing.assignment_dicts()
        _, fixed_targets = self._build_reception_assignment(state, [])
        stages = [
            AtLeastCompletedObjectivesStage(
                delivery_assignments=assignments[:2],
                minimum_completed_goals=1,
                instruction=parameters.initial_instruction,
                flag_answer=False,
            )
        ]
        for minimum in range(2, 4):
            stage = AtLeastCompletedObjectivesStage(
                delivery_assignments=assignments,
                minimum_completed_goals=minimum,
                instruction=parameters.added_instruction if minimum == 2 else "none",
                flag_answer=False,
            )
            if minimum == 2:
                stage.stage_input.linked_to_prev = True
            stages.append(stage)
        stages[0].entry_transition = ResetAdvancedDeliveryTransition(
            reception_objects=list(parameters.processing.current_reception_objects),
            broken_objects=list(fixed_targets),
            reception_offsets=parameters.processing.reception_offsets,
        )
        return stages

    def apply_request(
        self,
        state: TaskState,
        parameters: DeliveryInProgressParameters,
    ) -> TaskState:
        return save_assignments(state, parameters.processing.assignment_dicts())
