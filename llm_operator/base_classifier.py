from llama.program.util.run_ai import query_run_embedding

from llama import LlamaV2Runner

from sklearn.svm import OneClassSVM

from tqdm import tqdm

import re
import random
import pickle

import logging

logger = logging.getLogger(__name__)



class LlamaClassifier2:
    """A zero shot classifier that uses the Lamini LlamaV2Runner to generate
    examples from prompts and then uses a logistic regression to classify
    the examples.
    """

    def __init__(
        self,
        config: dict = {},
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        augmented_example_count: int = 10,
        generator_from_prompt=None,
        example_modifier=None,
        example_expander=None,
    ):
        self.config = config
        self.model_name = model_name
        self.augmented_example_count = augmented_example_count

        if generator_from_prompt is None:
            generator_from_prompt = DefaultExampleGenerator
        self.generator_from_prompt = generator_from_prompt

        if example_modifier is None:
            example_modifier = DefaultExampleModifier
        self.example_modifier = example_modifier

        if example_expander is None:
            example_expander = DefaultExampleExpander
        self.example_expander = example_expander

        # Examples is a dict of examples, where each row is a different
        # example class, followed by examples of that class
        self.examples = {}
        self.class_ids = {}

    def prompt_train(self, prompts: dict):
        """Trains the classifier using prompts for each class.

        First, augment the examples for each class using the prompts.
        """
        # Generate examples from prompts
        for class_name, prompt in prompts.items():
            self.add_class(class_name)
            generated_examples = self.generate_examples_from_prompt(
                prompt, self.examples.get(class_name, [])
            )

            #print(f"Generated {len(generated_examples)} examples for {class_name}", generated_examples)

            self.examples[class_name] = generated_examples

        self.train()

    def train(self):
        # Form the embeddings
        X = []
        y = []

        for class_name, examples in self.examples.items():
            print(f"Class {class_name} has {len(examples)} examples")
            index = self.class_ids[class_name]
            y += [index] * len(examples)
            class_embeddings = self.get_embeddings(examples)
            X += class_embeddings

        # Train the classifier
        self.logistic_regression = OneClassSVM(gamma='scale', nu=0.01).fit(X, y)

    def add_data_to_class(self, class_name, examples):
        self.add_class(class_name)
        if not class_name in self.examples:
            self.examples[class_name] = []
        self.examples[class_name] += examples

    def add_class(self, class_name):
        if not class_name in self.class_ids:
            self.class_ids[class_name] = len(self.class_ids)

    def get_data(self):
        return self.examples

    def get_embeddings(self, examples):
        embeddings = query_run_embedding(examples, config=self.config)

        return [embedding[0] for embedding in embeddings]

    def predict(self, text):
        class_id = self.logistic_regression.predict(self.get_embeddings(text))
        print("here:", class_id)
        # probs = self.predict_proba(text)
        #
        # # select the class with the highest probability, note that text and
        # # probs are lists of arbitrary length
        # winning_classes = [
        #     max(enumerate(prob), key=lambda x: x[1])[0] for prob in probs
        # ]

        # convert the class ids to class names
        class_name = list(self.class_ids.keys())[0]
        if class_id[0] == -1:
            return [class_name]
        else:
            return ["not"+class_name]

    def dumps(self):
        return pickle.dumps(self)

    @staticmethod
    def loads(data):
        return pickle.loads(data)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def create_new_example_generator(self, prompt, examples):
        example_generator = self.generator_from_prompt(
            prompt, examples, config=self.config, model_name=self.model_name
        )
        example_modifier = self.example_modifier(
            config=self.config, model_name=self.model_name
        )
        example_expander = self.example_expander(
            prompt, config=self.config, model_name=self.model_name
        )

        index = len(examples)

        while True:
            # Phase 1: Generate example types from prompt
            compressed_example_features = example_generator.generate_examples(
                seed=index
            )

            # Phase 2: Modify the features to be more diverse
            different_example_features = example_modifier.modify_examples(
                compressed_example_features
            )

            different_example_features += compressed_example_features

            # Phase 3: Expand examples from features
            for features in different_example_features:
                expanded_example = example_expander.expand_example(features)
                logger.debug(
                    f"Generated example number {index} out of {self.augmented_example_count}"
                )

                index += 1
                yield expanded_example

    def generate_examples_from_prompt(self, prompt, original_examples):
        examples = []

        for example in tqdm(
            self.create_new_example_generator(prompt, original_examples),
            total=self.augmented_example_count,
        ):
            examples.append(example)

            if len(examples) >= self.augmented_example_count:
                break

        #print("original_examples", original_examples)

        return examples + original_examples


class DefaultExampleGenerator:
    def __init__(
        self,
        prompt,
        examples,
        config=None,
        example_count=5,
        model_name="meta-llama/Llama-2-7b-chat-hf",
    ):
        self.prompt = prompt
        self.examples = examples.copy()
        self.config = config
        self.example_count = example_count
        self.model_name = model_name

        self.max_history = 2

    def generate_examples(self, seed):
        runner = LlamaV2Runner(config=self.config, model_name=self.model_name)

        prompt, system_prompt = self.get_prompt(seed=seed)

        logger.debug("+++++++ Default Example Generator Prompt ++++++++")
        logger.debug(prompt)
        logger.debug("+++++++ Default Example Generator System Prompt ++++++++")
        logger.debug(system_prompt)
        logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

        result = runner(prompt, system_prompt)

        logger.debug("+++++++ Default Example Generator Result ++++++++")
        logger.debug(result)
        logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

        examples = self.parse_result(result)

        return examples

    def get_prompt(self, seed):
        system_prompt = "You are a domain expert who is able to generate many different examples given a description."

        prompt = ""

        # Randomly shuffle the examples
        random.seed(seed)
        random.shuffle(self.examples)

        # Include examples if they are available
        if len(self.examples) > 0:
            selected_example_count = min(self.max_history, len(self.examples))

            prompt += "Consider the following examples:\n"

            for i in range(selected_example_count):
                prompt += "----------------------------------------\n"
                prompt += f"{self.examples[i]}"
                prompt += "\n----------------------------------------\n"

        prompt += "Read the following description carefully:\n"
        prompt += "----------------------------------------\n"
        prompt += self.prompt
        prompt += "\n----------------------------------------\n"

        prompt += f"Generate {self.example_count} different example summaries following this description. Each example summary should be as specific as possible using at most 10 words.  Start each example with a digit, e.g.\n 1. first example summary, 2. second example summary, etc.\n"

        return prompt, system_prompt

    def parse_result(self, result):
        results = re.findall(r"\d+\..*\n", result)

        # strip the numbers from the beginning of each result using the re module
        results = [re.sub(r"^\d+\.\s*", "", result) for result in results]

        if len(results) == 0:
            results = [result]

        return results


class DefaultExampleModifier:
    def __init__(self, config=None, model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.config = config
        self.model_name = model_name

    def modify_examples(self, examples):
        runner = LlamaV2Runner(config=self.config, model_name=self.model_name)

        prompt, system_prompt = self.get_prompt(examples)

        logger.debug("+++++++ Default Example Modifier Prompt ++++++++")
        logger.debug(prompt)
        logger.debug("+++++++ Default Example Modifier System Prompt ++++++++")
        logger.debug(system_prompt)
        logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

        result = runner(prompt, system_prompt)

        logger.debug("+++++++ Default Example Modifier Result ++++++++")
        logger.debug(result)
        logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

        examples = self.parse_result(result)

        return examples

    def get_prompt(self, examples):
        system_prompt = "You are a domain expert who is able to clearly understand these descriptions and modify them to be more diverse."

        prompt = "Read the following descriptions carefully:\n"
        prompt += "----------------------------------------\n"
        for index, example in enumerate(examples):
            prompt += f"{index + 1}. {example}\n"
        prompt += "\n----------------------------------------\n"

        prompt += "Generate 5 more examples that are similar, but substantially different from those. Each example should be as specific as possible using at most 10 words.  Start each example with a digit, e.g.\n 1. first example, 2. second example, etc.\n"

        return prompt, system_prompt

    def parse_result(self, result):
        results = re.findall(r"\d+\..*\n", result)

        # strip the numbers from the beginning of each result using the re module
        results = [re.sub(r"^\d+\.\s*", "", result) for result in results]

        if len(results) == 0:
            results = [result]

        return results


class DefaultExampleExpander:
    def __init__(self, prompt, config=None, model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.prompt = prompt
        self.config = config
        self.model_name = model_name

    def expand_example(self, example):
        runner = LlamaV2Runner(config=self.config, model_name=self.model_name)

        prompt, system_prompt = self.get_prompt(example)

        logger.debug("+++++++ Default Example Expander Prompt ++++++++")
        logger.debug(prompt)
        logger.debug("+++++++ Default Example Expander System Prompt ++++++++")
        logger.debug(system_prompt)
        logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

        result = runner(prompt, system_prompt)

        logger.debug("+++++++ Default Example Expander Result ++++++++")
        logger.debug(result)
        logger.debug("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

        return result

    def get_prompt(self, example):
        system_prompt = "You are a domain expert who is able to clearly understand this description and expand to a complete example from a short summary."

        prompt = "Read the following description carefully:\n"
        prompt += "----------------------------------------\n"
        prompt += self.prompt
        prompt += "\n----------------------------------------\n"
        prompt += "Now read the following summary of an example matching this description carefully:\n"
        prompt += "----------------------------------------\n"
        prompt += example
        prompt += "\n----------------------------------------\n"

        prompt += "Expand the summary to a complete example.  Be consistent with both the summary and the description.\n"

        return prompt, system_prompt
