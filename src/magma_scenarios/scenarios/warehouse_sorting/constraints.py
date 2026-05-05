from magma_scenarios.templates.constraints import RelationAssignmentConstraint


OBJECTS_AREA_KEY = "object_area"

class ObjectAreaContrait(RelationAssignmentConstraint) :
    def __init__(self, obj: str, area: str) -> None:
        super().__init__(
            source_value=obj,
            target_value=area,
            relation_key=OBJECTS_AREA_KEY,
            source_attribute_key="objects",
            target_attribute_key="target_areas",            
        )