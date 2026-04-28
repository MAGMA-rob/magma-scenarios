from typing import Optional

from magma_core.base.constraints import BaseConstraint
from magma_core.base.state.task_state import TaskState

class RelationAssignmentConstraint(BaseConstraint):
    """Assign one value to another in a configurable latent sorting relation.

    This generic constraint replaces specialized cases such as:
    - object -> area, stored in ``state.relations["object_area"]``
    - object -> category, stored in ``state.relations["object_type"]``
    - category -> area, stored in ``state.relations["type_area"]``

    Parameters
    ----------
    source_value:
        The key written on the left side of the relation.
        Example: ``"cup"`` in ``object_type["cup"] = "fragile"``.
    target_value:
        The value written on the right side of the relation.
        Example: ``"fragile"`` in ``object_type["cup"] = "fragile"``.
    relation_key:
        Name of the relation dictionary in ``state.relations``.
    source_attribute_key:
        Optional attribute list used to validate that ``source_value`` exists
        before applying the constraint.
        Example: ``"objects"`` when the source is an object name.
    target_attribute_key:
        Optional attribute list used to validate that ``target_value`` exists
        before applying the constraint.
        Example: ``"target_areas"`` when the target is an area name.

    Examples
    --------
    ``RelationAssignmentConstraint("cup", "zone_a", "object_area",
    source_attribute_key="objects", target_attribute_key="target_areas")``
        Validates both values, then writes the default area for the object.

    ``RelationAssignmentConstraint("cup", "fragile", "object_type",
    source_attribute_key="objects")``
        Validates that ``"cup"`` is a known object, then writes the category.

    ``RelationAssignmentConstraint("fragile", "zone_a", "type_area",
    target_attribute_key="target_areas")``
        Validates that ``"zone_a"`` is a known area, then writes the routing rule.
    """

    def __init__(
        self,
        source_value: str,
        target_value: str,
        relation_key: str,
        source_attribute_key: Optional[str] = None,
        target_attribute_key: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.source_value = source_value
        self.target_value = target_value
        self.relation_key = relation_key
        self.source_attribute_key = source_attribute_key
        self.target_attribute_key = target_attribute_key

    def _is_missing_attribute_value(
        self,
        state: TaskState,
        attribute_key: Optional[str],
        value: str,
    ) -> bool:
        """Return True when a value is expected in an attribute list but missing."""
        if attribute_key is None:
            return False
        return value not in state.attributes.get(attribute_key, [])

    def apply(self, state: TaskState):
        super().apply(state) # important for register the constraint in the state list.
        # Each side of the relation can be validated independently.
        # This is why the class accepts two optional attribute keys.
        if self._is_missing_attribute_value(state, self.source_attribute_key, self.source_value):
            raise RuntimeError(f"The {self.__class__.__name__} failed to be applied")
        if self._is_missing_attribute_value(state, self.target_attribute_key, self.target_value):
            raise RuntimeError(f"The {self.__class__.__name__} failed to be applied")
        if self.relation_key not in state.relations:
            state.relations[self.relation_key] = {}
        state.relations[self.relation_key][self.source_value] = self.target_value

    def outdated(self, state: TaskState) -> bool:
        # A constraint becomes outdated if one of the values it depends on
        # is no longer present in the corresponding validated attribute list.
        return (
            self._is_missing_attribute_value(state, self.source_attribute_key, self.source_value)
            or self._is_missing_attribute_value(state, self.target_attribute_key, self.target_value)
        )
