from collections import defaultdict
from dataclasses import asdict, dataclass, field
import traceback
from warnings import warn
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str
from langchain.schema import HumanMessage, SystemMessage

from agents.base import Agent
# from agents import dynamic_prompting
from agents import my_prompting
from agents.prompt_utils import prune_html
from utils.llm_utils import ParseError, retry
from utils.chat_api import ChatModelArgs

import itertools
import numpy as np
from tqdm import trange
import math


@dataclass
class MyLwmAgentArgs:
    agent_name: str = "MyLwmAgent"
    chat_model_args: ChatModelArgs = None
    flags: my_prompting.Flags = field(default_factory=lambda: my_prompting.Flags())
    max_retry: int = 4

    def make_agent(self):
        return MyLwmAgent(
            chat_model_args=self.chat_model_args, flags=self.flags, max_retry=self.max_retry
        )


class MyLwmAgent(Agent):
    def __init__(
        self,
        chat_model_args: ChatModelArgs = None,
        flags: my_prompting.Flags = None,
        max_retry: int = 4,
        **kwargs,
    ):
        if chat_model_args is None:
            chat_model_args = ChatModelArgs()
        self.chat_llm = chat_model_args.make_chat_model()
        self.chat_llm.temperature = 1.0
        self.chat_llm_high_temp = self.chat_llm
        
        # chat_model_args.n = 6
        # self.chat_llm_sample = chat_model_args.make_chat_model()
        
        # chat_model_args.temperature = 1.9
        # self.chat_llm_high_temp = chat_model_args.make_chat_model()
        
        self.chat_model_args = chat_model_args
        self.max_retry = max_retry

        if flags is None:
            self.flags = my_prompting.Flags()
        else:
            self.flags = flags
        
        # calling this just in case, but it should be called by benchmark before the first step
        self.reset(seed=None)

        if kwargs:
            warn(f"Warning: Not using any of these arguments when initiating the agent: {kwargs}")

    def encoder(self, main_prompt): 
        state_prompt = main_prompt.get_encoder_prompt()

        chat_messages = [
            SystemMessage(content=my_prompting.SystemPrompt().prompt),
            HumanMessage(content=state_prompt),
        ]

        def parser(text):
            try:
                ans_dict = main_prompt._parse_state_answer(text)
            except ParseError as e:
                # these parse errors will be caught by the retry function and
                # the chat_llm will have a chance to recover
                return None, False, str(e)

            return ans_dict, True, ""

        try:
            ans_dict = retry(self.chat_llm, chat_messages, n_retry=self.max_retry, parser=parser)
            # inferring the number of retries, TODO: make this less hacky
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ValueError as e:
            # Likely due to maximum retry. We catch it here to be able to return
            # the list of messages for further analysis
            ans_dict = {"state": None}
            ans_dict["err_msg"] = str(e)
            ans_dict["stack_trace"] = traceback.format_exc()
            ans_dict["n_retry"] = self.max_retry

        ans_dict["chat_messages"] = [m.content for m in chat_messages]
        ans_dict["chat_model_args"] = asdict(self.chat_model_args)
        ans_dict["prompt"] = state_prompt

        return ans_dict["state"], ans_dict


    def policy(self, main_prompt): 
        # Determine the minimum non-None token limit from prompt, total, and input tokens, or set to None if all are None.
        maxes = (
            # self.flags.max_prompt_tokens,
            self.chat_model_args.max_total_tokens,
            self.chat_model_args.max_input_tokens,
        )
        maxes = [m for m in maxes if m is not None]
        max_prompt_tokens = min(maxes) if maxes else None

        prompt = my_prompting.fit_tokens(
            main_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
        )

        chat_messages = [
            SystemMessage(content=my_prompting.SystemPrompt().prompt),
            HumanMessage(content=prompt),
        ]

        def parser(text):
            try:
                ans_dict = main_prompt._parse_answer(text)
            except ParseError as e:
                # these parse errors will be caught by the retry function and
                # the chat_llm will have a chance to recover
                return None, False, str(e)

            return ans_dict, True, ""

        try:
            ans_dict = retry(self.chat_llm_high_temp, chat_messages, n_retry=self.max_retry, parser=parser)
            # inferring the number of retries, TODO: make this less hacky
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ValueError as e:
            # Likely due to maximum retry. We catch it here to be able to return
            # the list of messages for further analysis
            ans_dict = {"action": None}
            ans_dict["err_msg"] = str(e)
            ans_dict["stack_trace"] = traceback.format_exc()
            ans_dict["n_retry"] = self.max_retry

        ans_dict["chat_messages"] = [m.content for m in chat_messages]
        ans_dict["chat_model_args"] = asdict(self.chat_model_args)
        ans_dict["prompt"] = main_prompt._prompt

        return ans_dict["action"], ans_dict

    
    def dynamics(self, main_prompt): 
        dynamics_prompt = main_prompt.get_dynamics_prompt()

        chat_messages = [
            SystemMessage(content=my_prompting.SystemPrompt().prompt),
            HumanMessage(content=dynamics_prompt),
        ]

        def parser(text):
            try:
                ans_dict = main_prompt._parse_dynamics_answer(text)
            except ParseError as e:
                # these parse errors will be caught by the retry function and
                # the chat_llm will have a chance to recover
                return None, False, str(e)

            return ans_dict, True, ""

        try:
            ans_dict = retry(self.chat_llm, chat_messages, n_retry=self.max_retry, parser=parser)
            # inferring the number of retries, TODO: make this less hacky
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ValueError as e:
            # Likely due to maximum retry. We catch it here to be able to return
            # the list of messages for further analysis
            ans_dict = {"next_state": None,
                        "status": None}
            ans_dict["err_msg"] = str(e)
            ans_dict["stack_trace"] = traceback.format_exc()
            ans_dict["n_retry"] = self.max_retry

        ans_dict["chat_messages"] = [m.content for m in chat_messages]
        ans_dict["chat_model_args"] = asdict(self.chat_model_args)
        ans_dict["prompt"] = dynamics_prompt

        is_terminal = (ans_dict['status'] == 'goal-reached')

        return ans_dict["next_state"], is_terminal, ans_dict

    def action_reward(self, main_prompt): 
        action_reward_prompt = main_prompt.get_action_reward_prompt()

        chat_messages = [
            SystemMessage(content=my_prompting.SystemPrompt().prompt),
            HumanMessage(content=action_reward_prompt),
        ]

        def parser(text):
            try:
                ans_dict = main_prompt._parse_action_reward_answer(text)
            except ParseError as e:
                # these parse errors will be caught by the retry function and
                # the chat_llm will have a chance to recover
                return None, False, str(e)

            return ans_dict, True, ""

        try:
            ans_dict = retry(self.chat_llm, chat_messages, n_retry=self.max_retry, parser=parser)
            # inferring the number of retries, TODO: make this less hacky
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ValueError as e:
            # Likely due to maximum retry. We catch it here to be able to return
            # the list of messages for further analysis
            ans_dict = {"response": None}
            ans_dict["err_msg"] = str(e)
            ans_dict["stack_trace"] = traceback.format_exc()
            ans_dict["n_retry"] = self.max_retry

        ans_dict["chat_messages"] = [m.content for m in chat_messages]
        ans_dict["chat_model_args"] = asdict(self.chat_model_args)
        ans_dict["prompt"] = action_reward_prompt

        response = ans_dict["response"]
        reward = -1 if response == 'away-from-the-goal' else 1 if response == 'towards-the-goal' else 0

        return reward, ans_dict

    # def critic(self, main_prompt): 
    #     evaluation_prompt = main_prompt.get_critic_prompt()

    #     chat_messages = [
    #         SystemMessage(content=my_prompting.SystemPrompt().prompt),
    #         HumanMessage(content=evaluation_prompt),
    #     ]

    #     def parser(text):
    #         try:
    #             ans_dict = main_prompt._parse_evaluation_answer(text)
    #         except ParseError as e:
    #             # these parse errors will be caught by the retry function and
    #             # the chat_llm will have a chance to recover
    #             return None, False, str(e)

    #         return ans_dict, True, ""

    #     try:
    #         ans_dict = retry(self.chat_llm, chat_messages, n_retry=self.max_retry, parser=parser)
    #         # inferring the number of retries, TODO: make this less hacky
    #         ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
    #     except ValueError as e:
    #         # Likely due to maximum retry. We catch it here to be able to return
    #         # the list of messages for further analysis
    #         ans_dict = {"evaluation": None}
    #         ans_dict["err_msg"] = str(e)
    #         ans_dict["stack_trace"] = traceback.format_exc()
    #         ans_dict["n_retry"] = self.max_retry

    #     ans_dict["chat_messages"] = [m.content for m in chat_messages]
    #     ans_dict["chat_model_args"] = asdict(self.chat_model_args)
    #     ans_dict["prompt"] = next_state_prompt

    #     return ans_dict["evaluation"], ans_dict


    def get_action(self, obs):
        if not "pruned_html" in obs:
            obs["pruned_html"] = prune_html(obs["dom_txt"])

        self.obs_history.append(obs)
        main_prompt = my_prompting.MyMainPrompt(
            obs_history=self.obs_history,
            states=self.states,
            actions=self.actions
        )

        state, ans_dict = self.encoder(main_prompt)
        print('State:', state)

        # main_prompt = my_prompting.MyMainPrompt(
        #     obs_history=self.obs_history,
        #     states=self.states,
        #     actions=self.actions
        # )

        # print('Prompt:')
        # print(main_prompt._prompt)

        # Run MCTS Search
        class MCTSNode(): 
            id_iter = itertools.count()
            
            @classmethod
            def reset_id(cls):
                cls.id_iter = itertools.count()
                
            def __init__(self, state=None, action=None, parent=None, 
                         fast_reward=0, is_terminal=False): 
                self.state = state
                self.state_dict = None
                self.strategy = None
                self.action = action
                self.action_dict = None
                self.fast_reward = self.reward = fast_reward
                self.fast_reward_dict = None
                self.reward_dict = None
                self.cum_rewards = []
                self.parent = parent
                self.children = None
                self.is_terminal = is_terminal
                if parent is None: 
                    self.depth = 0
                else: 
                    self.depth = parent.depth + 1
                    
            @property
            def Q(self):
                return np.mean(self.cum_rewards)

        def _expand(node, path): 
            new_states = [n.state for n in path[:] if n.state is not None]
            new_actions = [n.action for n in path[:] if n.action is not None]
            if node.state is None: 
                
                # print(self.states + new_states)
                # print(self.actions + new_actions)
                main_prompt = my_prompting.MyMainPrompt(
                    obs_history=self.obs_history,
                    states=self.states + new_states,
                    actions=self.actions + new_actions
                )
                node.state, node.is_terminal, node.state_dict = self.dynamics(main_prompt)
                print('Next State:', node.state)
                print('Status:', node.state_dict['status'])
                new_states.append(node.state)

                # Here is a chance to reset the node reward using things like state transition certainty
                # or state-conditional critic (value function)
                # As a default we just keep using the fast reward
                node.reward, node.reward_dict = node.fast_reward, node.fast_reward_dict
                
                # main_prompt = my_prompting.MyMainPrompt(
                #     obs_history=self.obs_history,
                #     states=self.states + new_states,
                #     actions=self.actions + new_actions
                # )
                # node.reward, node.is_terminal, node.reward_details = self.critic(main_prompt)
                # TODO (DONE) : Figure out numerical reward logic
            if not node.is_terminal: 
                children = []
                # Sample an action space:
                action_space = {}
                n_actions = 2
                for i in range(n_actions): 
                    main_prompt = my_prompting.MyMainPrompt(
                        obs_history=self.obs_history,
                        states=self.states + new_states,
                        actions=self.actions + new_actions
                    )
                    action, action_dict = self.policy(main_prompt)
                    action_space[action] = action_dict
                    
                for action, action_dict in action_space.items(): 
                    # TODO (DONE): Figrue out how fast reward is computed
                    # fast_reward = 0
                    # print(self.states + new_states)
                    # print(self.actions + new_actions)
                    main_prompt = my_prompting.MyMainPrompt(
                        obs_history=self.obs_history,
                        states=self.states + new_states,
                        actions=self.actions + new_actions + [action]
                    )
                    fast_reward, fast_reward_dict = self.action_reward(main_prompt)
                    print('Strategy Candidate:', action_dict['strategy'])
                    print('Action Candidate:', action)
                    print('Fast Reward:', fast_reward)
                    child = MCTSNode(state=None, action=action, parent=node,
                                     fast_reward=fast_reward)
                    child.action_dict = action_dict
                    child.fast_reward_dict = fast_reward_dict
                    children.append(child)
                node.children = children

        w_exp = 1
        depth_limit = 3
        def _uct(node): 
            print(node.Q)
            print(np.sqrt(np.log(len(node.parent.cum_rewards)) / max(1, len(node.cum_rewards))))
            return node.Q + w_exp * np.sqrt(np.log(len(node.parent.cum_rewards)) / max(1, len(node.cum_rewards)))
        # _uct = lambda node: node.Q + w_exp * np.sqrt(np.log(len(node.parent.cum_rewards)) / max(1, len(node.cum_rewards)))
        _is_terminal_with_depth_limit = lambda node: node.is_terminal or node.depth >= depth_limit

        N = 3
        root = MCTSNode(state=state, action=None, parent=None)
        # for n in trange(N, desc='MCTS iteration', leave=True): 
        for n in range(N): 
            print('MCTS iter', n)
            # select
            node = root
            path = []
            finished = False
            while not finished: 
                path.append(node)
                if (node.children is None or len(node.children) == 0 
                    or _is_terminal_with_depth_limit(node)): 
                    finished = True
                else: 
                    # uct select with fast reward
                    node = max(node.children, key=_uct)

            node = path[-1]
            if not _is_terminal_with_depth_limit(node): 
                # expand
                _expand(node, path)
                # simulate
                finished = False
                while not finished: 
                    if node.state is None: 
                        _expand(node, path)
                    if _is_terminal_with_depth_limit(node) or len(node.children) == 0: 
                        finished = True
                    else:
                        fast_rewards = [child.fast_reward for child in node.children]
                        # TODO (DONE): Simulate choice
                        node = node.children[np.argmax(fast_rewards)]
                        path.append(node)
            # backpropagate
            rewards = []
            cum_reward = -math.inf
            for node in reversed(path): 
                rewards.append(node.reward)
                cum_reward = np.sum(rewards[::-1])
                node.cum_rewards.append(cum_reward)

        # max reward output strategy
        # dfs on max reward
        def _dfs_max_reward(path): 
            cur = path[-1]
            if cur.is_terminal: 
                return sum([node.reward for node in path[1:]]), path
            if cur.children is None: 
                return -math.inf, path
            visited_children = [x for x in cur.children if x.state is not None]
            if len(visited_children) == 0: 
                return -math.inf, path
            return max((_dfs_max_reward(path + [child]) for child in visited_children), key=lambda x: x[0])
        output_cum_reward, output_iter = _dfs_max_reward([root])

        action, action_dict = output_iter[1].action, output_iter[1].action_dict
        print('Selected Strategy:', action_dict['strategy'])
        print('Selected Action:', action)
        self.states.append(state)
        self.actions.append(action)

        return action, action_dict


    def reset(self, seed=None):
        self.seed = seed
        self.actions = []
        self.obs_history = []
        self.states = []
        self.evaluations = []
        self.strategies = []


    def preprocess_obs(self, obs: dict) -> dict:
        obs["dom_txt"] = flatten_dom_to_str(
            obs["dom_object"],
            with_visible=self.flags.extract_visible_tag,
            with_center_coords=self.flags.extract_coords == "center",
            with_bounding_box_coords=self.flags.extract_coords == "box",
            filter_visible_only=self.flags.extract_visible_elements_only,
        )

        obs["axtree_txt"] = flatten_axtree_to_str(
            obs["axtree_object"],
            with_visible=self.flags.extract_visible_tag,
            with_center_coords=self.flags.extract_coords == "center",
            with_bounding_box_coords=self.flags.extract_coords == "box",
            filter_visible_only=self.flags.extract_visible_elements_only,
        )

        obs["pruned_html"] = prune_html(obs["dom_txt"])

    def get_action_mapping(self) -> callable:
        return my_prompting._get_my_action_space().to_python_code
