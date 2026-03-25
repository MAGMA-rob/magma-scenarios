from typing import Dict

from magma_core.base.data_structures import TemplateInstruction

att = {"product_type":["coca","icetea","brets","donut"]}

MAX_NB_PER_RECIPE = 2
    
class DeliveryTemplateInstruction(TemplateInstruction):

    def __init__(self, template: Dict, timestamp: int = 0) -> None:
        context = """
        You will have access to a dict with add and remove field. 
        You must generate an instruction that inform the robot hat the default recipe for its packaging has changed. 
        The add field represent element that need to be added to the recipe, remove element represent elements that need to be removed from the recipe.
        If an element is present in both, you can safely ignore it. If there is x time the same element in the same field, you can just tell the robot x element. 
        You are only giving a constraint, you MUST NOT ask to launch a cycle with this recipe, just update it for future cycle.
        Here are some exemple 'add x to your recipe for future cycle', 'remove y and add x from your recipe now' ...
        """
        super().__init__(template, context, timestamp)
