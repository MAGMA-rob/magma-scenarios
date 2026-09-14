from magma_scenarios.templates.requests import (
    GiveCategoryAssignmentRequest,
    GiveObjectAssignmentRequest,
    GiveObjectCategoryRequest,
)


class WarehouseObjectAssignmentRequest(GiveObjectAssignmentRequest):
    pass


class WarehouseObjectCategoryRequest(GiveObjectCategoryRequest):
    pass


class WarehouseCategoryAssignmentRequest(GiveCategoryAssignmentRequest):
    pass
