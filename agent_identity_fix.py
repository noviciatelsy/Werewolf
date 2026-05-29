
    def __call__(self, args, observation: List[Message], messages: MessagePool, questions: QuestionPool,
                 state=(0, "daytime", "", [])) -> str:
        """
        Call the agents to generate a response (equivalent to taking an action).
        state = (turn, phase, role, alive_list)
        """
        try:
            if self.name == "Player 1" and args and hasattr(args, 'human_in_combat') and args.human_in_combat:
                response = input("Now you say: ")
                with open(os.path.join(args.logs_path_to, str(args.current_game_number) + ".md"), "a") as f:
                    f.write(f"Player 1: {response}  " + "\n")
            else:
                # 关键修复：确保role参数正确传递，并添加身份验证
                current_role = state[2] if len(state) > 2 else ""
                
                # 添加身份验证逻辑
                if current_role and current_role != "unknown":
                    # 在prompt中明确当前agent的身份
                    enhanced_prompt = f"You are {self.name}, and you are a {current_role}. Remember this throughout the conversation."
                    if self.global_prompt:
                        self.global_prompt = enhanced_prompt + "\n" + self.global_prompt
                    else:
                        self.global_prompt = enhanced_prompt
                
                response = self.backend.query(args, agent_name=self.name, role_desc=self.role_desc,
                                            history_messages=observation, global_prompt=self.global_prompt,
                                            request_msg=None, msgs=messages, ques=questions, turns=state[0],
                                            day_night=state[1], role=current_role, alives=state[3])
        except RetryError as e:
            logging.warning(f"Agent {self.name} failed to generate a response. "
                            f"Error: {e.last_attempt.exception()}. "
                            f"Sending signal to end the conversation.")
            response = SIGNAL_END_OF_CONVERSATION
        except Exception as e:
            logging.warning(f"Agent {self.name} failed to generate a response. "
                            f"Error: {e}. Sending signal to end the conversation.")
            response = SIGNAL_END_OF_CONVERSATION

        return response
    