import torch

BTN_STROKE = 0.011

def tensor_is_button_pressed(btn_translation) -> torch.Tensor:
    button_STROKE_LIMIT = -BTN_STROKE/2 * torch.ones(btn_translation.size())
    return (btn_translation < button_STROKE_LIMIT.to(btn_translation.device))

attributes = {"objects": ["sw0","sw1","sw2","sw3","sw4"]}