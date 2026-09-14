from typing import Any, Dict, List

from magma_core.simulation.data_structures import ToolResult

from ..common.errors import MaskItemsError


class AdvancedMaskItemsError(MaskItemsError):
    def _to_spec_arguments(self) -> Dict[str, Any]:
        return {
            "max_masking": self.max_nb,
            "required_type_counts": self.required_type_counts,
            "fixed_targets_by_object": self.fixed_targets_by_object,
            "minimum": self.minimum,
            "thresh": self.thresh,
        }

    def apply_post_verif(
        self,
        tool_result: ToolResult,
        arguments: Dict[str, Any],
    ) -> None:
        if not tool_result.context:
            return

        masked = set(arguments.get("masked", []))
        if not masked:
            return

        objects_by_location: Dict[str, Dict[str, Any]] = (
            tool_result.context["objects_by_location"]
        )
        dirty_objects: List[str] = tool_result.context["dirty_objects"]
        visible_dirty_objects = [
            name for name in dirty_objects
            if name not in masked
        ]

        tool_result.context = objects_by_location
        super().apply_post_verif(tool_result, arguments)
        visible_by_location = tool_result.context

        if not visible_dirty_objects:
            tool_result.reason += " There are no visible dirty objects."
        elif len(visible_dirty_objects) == 1:
            tool_result.reason += (
                f" Dirty object is {visible_dirty_objects[0]}."
            )
        else:
            tool_result.reason += (
                f" Dirty objects are {', '.join(visible_dirty_objects)}."
            )

        tool_result.context = {
            "objects_by_location": visible_by_location,
            "dirty_objects": visible_dirty_objects,
        }
