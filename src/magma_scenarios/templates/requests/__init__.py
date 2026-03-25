from .assignment_requests import (
    GiveObjectAssignmentRequest,
    GiveObjectCategoryRequest,
    GiveCategoryAssignmentRequest
)

from .attributes_request import (
    AddValueToListRequest,
    RemoveValueToListRequest
)

__all__ = ["GiveObjectAssignmentRequest","GiveObjectCategoryRequest","GiveCategoryAssignmentRequest", 
           "AddValueToListRequest", "RemoveValueToListRequest"]