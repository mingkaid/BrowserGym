from dataclasses import dataclass, field
from utils.chat_api import ChatModelArgs


@dataclass
class AgentArgs:
    agent_name: str
    chat_model_args: ChatModelArgs = None
    kwargs: dict = field(default_factory=dict)

    def make_agent(self):
        match self.agent_name:
            case "GenericAgent":
                from agents.generic_agent import GenericAgent

                return GenericAgent(chat_model_args=self.chat_model_args, **self.kwargs)
                
            case "MyVanillaAgent":
                from agents.my_vanilla_agent import MyVanillaAgent

                return MyVanillaAgent(chat_model_args=self.chat_model_args, **self.kwargs)

            case "MyStateAgent":
                from agents.my_state_agent import MyStateAgent

                return MyStateAgent(chat_model_args=self.chat_model_args, **self.kwargs)
                
            case _:
                raise ValueError(f"agent_name {self.agent_name} not recognized")
