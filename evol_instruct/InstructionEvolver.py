from evol_instruct.init.logger import logger
import random

class InstructionEvolution:
    def __init__(self, initial_instructions, config=None):
        self.pool = initial_instructions
        self.evolved_dataset = {
            'instruction': [],
            'response': [],
            'category': [],
            'evolution_strategy': [],
            'in-depth-evolving_operation': [],
            'epoch': []

        }
        self.config = {
            "strategy": None,
            "in_depth_evolution_operation": None,
            "prompt": None
        }

    def select_evolution_strategy(self):
        logger.info("Selecting evolution strategy")
        strategies = [
            (0, "in-depth-evolving"),
            (1, "in-breadth-evolving"),
        ]

        self.config["strategy"] = random.choice(strategies)
        logger.info(f"Evolution strategy: {self.config['strategy'][1]}")

        return self

    def select_in_depth_evolution_operation(self):
        logger.info("Selecting in-depth evolution operation")
        operations = [
            (0, "add-constraints"),
            (1, "deepening"),
            (2, "concretizing"),
            (3, "increase-reasoning-steps")
        ]

        self.config["in_depth_evolution_operation"] = random.choice(operations)
        logger.info(f"In-depth evolution operation: {self.config['in_depth_evolution_operation'][1]}")

        return self

    def format_prompt_with_in_depth_evolution_operation(self):
        match self.config['in_depth_evolution_operation'][0]:
            case 0:
                self.config["prompt"] = self.config["prompt"].format(operation="by adding one more constraints/requirements into #Given Prompt#")
            case 1:
                self.config["prompt"] = self.config["prompt"].format(operation="if #Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased.")
            case 2:
                self.config["prompt"] = self.config["prompt"].format(operation="by replacing general concepts with more specific concepts.")
            case 3:
                self.config["prompt"] = self.config["prompt"].format(operation="if #Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.")

        return self

    def generate_prompt(self, instruction):
        match self.config["strategy"][0]:
            case 0:
                self.config["prompt"] = """<human>: I want you to act as a prompt rewriter.
Your objective is to rewrite the #Given Prompt# into a more complex version.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:. Also, please do not omit the context in #Given Prompt#.
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #Given Prompt#.
‘#Given Prompt#’, ‘#Rewritten Prompt#’, ‘given prompt’ and ‘rewritten prompt’ are not allowed to appear in #Rewritten Prompt#
You SHOULD complicate the given prompt {operation}
#Given Prompt#:
{instruction}
<bot>: #Rewritten Prompt#:"""

                self.config["prompt"] = self.config["prompt"].format(operation="{operation}", instruction=instruction)
                self.format_prompt_with_in_depth_evolution_operation()
            case 1:
                self.config["prompt"] = """<human>: I want you to act as a prompt creator.
Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.
This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.
The LENGTH and difficulty level of the #Created Prompt# should be similar to that of the #Given Prompt#.
The #Created Prompt# must be reasonable and must be understood and responded by humans.
‘#Given Prompt#’, ‘#Created Prompt#’, ‘given prompt’ and ‘created prompt’ are not allowed to appear in #Created Prompt#.
Your response only contains the #Created Prompt# and no explanation of the new prompt. Do not provide a response to either the #Given Prompt# or the #Created Prompt#.
#Given Prompt#:
{instruction}
<bot>: #Created Prompt#:"""

                self.config["prompt"] = self.config["prompt"].format(instruction=instruction)

        print(f"Prompt: {self.config['prompt']}")
        return self

    def example_generator(self, generate):
        logger.info("Generating example")
        instruction = generate(self.config["prompt"]).replace("#Rewritten Prompt#:", "").replace("#Created Prompt#:", "").strip().strip('\n')
        print(instruction)

        logger.info("Generating response")
        response = generate(f"<human>: {instruction}\n<bot>: ")
        print(response)

        return instruction, response

    def instruction_evolver(self, instruction, generate):
        return self \
            .generate_prompt(instruction) \
            .example_generator(generate)


    def has_instruction_evolved(self, original_instruction, evolved_instruction, response, generate):

        def has_information_gain(original_instruction, evolved_instruction, generate, counter=0):
            if counter > 5:
                return False

            equality_check_prompt = f"""<human>: Do you think the following two instructions are equal to each other in that they meet the following requirements:
1. They have same constraints and requirements.
2. They have same depth and breadth of the inquiry.
The First Prompt: {original_instruction}
The Second Prompt: {evolved_instruction}
Your response should be either equal or not equal.
<bot>: The two prompts are """
            print(equality_check_prompt)
            model_output = generate(equality_check_prompt, temp=0.0).lower().replace("*", "")
            print(model_output)

            if "not equal" in model_output:
                return True
            elif "equal" in nltk.word_tokenize(model_output):
                return False
            else:
                return has_information_gain(original_instruction, evolved_instruction, generate, counter+1)


        def is_response_difficult(response):
            return 'sorry' in response and len(nltk.word_tokenize(response)) < 80

        def contains_only_punctuation_and_stop_words(response):
            stop_words = set(stopwords.words('english'))
            words = nltk.word_tokenize(response)
            return all(word in stop_words or word in string.punctuation for word in words)

        def contains_disallowed_phrases(instruction):
            disallowed_phrases = [
                "#Given Prompt#", "#Created Prompt#", "#Rewritten Prompt#",
                "given prompt", "created prompt", "rewritten prompt"]

            return any(phrase in instruction for phrase in disallowed_phrases)

        if \
        has_information_gain(original_instruction, evolved_instruction, generate) and \
        not is_response_difficult(response) and \
        not contains_only_punctuation_and_stop_words(response) and \
        not contains_disallowed_phrases(evolved_instruction):
            return True
        else:
            return False

    def save_dataset(self, epoch, category, file_name_manual_epoch="", file_name_append_tag=""):
        filename = os.path.join(
            "evolved",
            category,
            f"""{epoch if not file_name_manual_epoch else file_name_manual_epoch}_{self.config['strategy'][1]}{f"_{self.config['in_depth_evolution_operation'][1]}" if self.config['in_depth_evolution_operation'] else ''}{f"_{file_name_append_tag}" if file_name_append_tag else ""}.json""")

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        with open(filename, "w") as f:
            json.dump(self.evolved_dataset, f)


    def check_and_save_dataset(self, epoch, category, file_name_manual_epoch="", file_name_append_tag=""):
        global time0

        if len(self.evolved_dataset['instruction']) % 5 >= 0 or time() - time0 >= 300:
            logger.info("Saving...")

            self.save_dataset(epoch, category, file_name_manual_epoch, file_name_append_tag)
            time0 = time()
            return True
        else:
            print(time() - time0)
            return False

    def clear_evolved_instructions(self):
        self.evolved_dataset = {
            'instruction': [],
            'response': [],
            'category': [],
            'evolution_strategy': [],
            'in-depth-evolving_operation': [],
            'epoch': []

        }


    def evolve(self, example_generate, eval_generate, category, file_name_manual_epoch="", file_name_append_tag=""):
        for epoch in tqdm(range(NUM_EPOCHS), desc="Evolving", unit="epoch"):
            new_pool = []

            self.select_evolution_strategy()
            if self.config["strategy"][0] == 0:
                self.select_in_depth_evolution_operation()

            for instruction in tqdm(self.pool, desc="Instruction", unit="instruction"):
                try:
                    evolved_instruction, response = self.instruction_evolver(instruction, example_generate)
                    if self.has_instruction_evolved(instruction, evolved_instruction, response, eval_generate):
                        logger.info("Instruction Evolved")
                        print(f"Instruction Evolved: {evolved_instruction}\n\nResponse: {response}")

                        self.evolved_dataset['instruction'].append(evolved_instruction)
                        self.evolved_dataset['response'].append(response)
                        self.evolved_dataset['category'].append(category)
                        self.evolved_dataset['evolution_strategy'].append(self.config["strategy"][1])
                        if self.config["in_depth_evolution_operation"]:
                            self.evolved_dataset['in-depth-evolving_operation'].append(self.config["in_depth_evolution_operation"][1])
                        else:
                            self.evolved_dataset['in-depth-evolving_operation'].append("")
                        self.evolved_dataset['epoch'].append(epoch)

                        new_pool.append(evolved_instruction)

                        saved = self.check_and_save_dataset(epoch, category, file_name_manual_epoch, file_name_append_tag)
                        if saved:
                            logger.info("Saved")
                    else:
                        logger.info("Instruction Not Evolved")
                        print(f"Instruction Not Evolved: {evolved_instruction}")

                        new_pool.append(instruction)

                        with open("unevolved_instructions.txt", "a") as f:
                            f.write("------------------------------------------------------------------------------\n")
                            f.write(f"{epoch}, {category}\n")
                            f.write("Instruction Not Evolved\n")
                            f.write("------------------------------------------------------------------------------\n")

                            f.write(f"{instruction}\n")
                            f.write("========================================\n")
                            f.write(f"{evolved_instruction}\n")
                            f.write("========================================\n")
                            f.write(f"{response}\n")
                            f.write("\n\n\n")

                    clear_output(wait=True)
                except:
                    pass

            self.save_dataset(epoch, category, file_name_manual_epoch, file_name_append_tag)
            self.pool = new_pool
            self.clear_evolved_instructions()