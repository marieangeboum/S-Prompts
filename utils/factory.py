from methods.sprompt import SPrompts
from methods.base import BaseLearner
from methods.sprompt_spec import SPrompts_Spec
def get_model(model_name, args):
    name = model_name.lower()
    options = {'sprompts': SPrompts,
               'base': BaseLearner,
               'sprompts-spec' : SPrompts_Spec
               }
    return options[name](args)

