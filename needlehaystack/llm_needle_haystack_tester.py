import asyncio
import glob
import json
import os
import time

import numpy as np

from .evaluators import Evaluator
from .providers import ModelProvider
from .evaluation_context import EvaluationContext

from asyncio import Semaphore
from datetime import datetime, timezone

from .providers.model import TokenTextPair


class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """

    SENTENCE_ENDINGS = (".", "?", "!")
    SENTENCE_TRAILING_CHARS = set('"\'”’)]}›»')

    def __init__(self,
                 model_to_test: ModelProvider = None,
                 evaluator: Evaluator = None,
                 needle = None,
                 haystack_dir = "PaulGrahamEssays",
                 retrieval_question = None,
                 results_version = 1,
                 context_lengths_min = 1000,
                 context_lengths_max = 16000,
                 context_lengths_num_intervals = 35,
                 context_lengths = None,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 35,
                 document_depth_percents = None,
                 document_depth_percent_interval_type = "linear",
                 num_concurrent_requests = 1,
                 save_results = True,
                 overwrite_results = False,
                 save_contexts = True,
                 final_context_length_buffer = 200,
                 seconds_to_sleep_between_completions = None,
                 print_ongoing_status = True,
                 **kwargs):
        """
        :model_to_test: The model to test. Default is None.
        :evaluator: An evaluator to evaluate the model's response. Default is None.
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        :param kwargs: Additional arguments.
        """
        if not model_to_test:
            raise ValueError("A language model must be provided to test.")
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")

        self.needle = needle
        if self.needle[-1] != '.': # Make sure the needle ends with a period.
            self.needle += '.'
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.overwrite_results = overwrite_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.testing_results = []

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")

            if document_depth_percent_interval_type == 'linear':
                self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
            elif document_depth_percent_interval_type == 'sigmoid':
                self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
            else:
                raise ValueError("document_depth_percent_interval_type must be either 'sigmoid' or 'linear' if document_depth_percents is None.")
        else:
            self.document_depth_percents = document_depth_percents

        self.model_to_test = model_to_test
        self.model_name = self.model_to_test.model_name

        self.evaluation_model = evaluator

    def logistic(self, x, L=100, x0=50, k=.1):
        if x in [0, 100]:
            return x
        x = -k * (x - x0)
        return np.round(L * self.sigmoid(x), 3)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    async def bound_evaluate_and_log(self, sem, *args):
        async with sem:
            await self.evaluate_and_log(*args)

    async def run_test(self):
        sem = Semaphore(self.num_concurrent_requests)

        # Run through each iteration of context_lengths and depths
        tasks = []
        for context_length in self.context_lengths:
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(sem, context_length, depth_percent)
                tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

    async def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results and not self.overwrite_results:
            if self.result_exists(context_length, depth_percent):
                print("Warning! Skipping evaluation - the result already exists for context length ", context_length, " and depth percent ", depth_percent)
                return

        # Prepare evaluation context
        evaluation_context = EvaluationContext(
            model=self.model_name,
            context_length=int(context_length),
            depth_percent=float(depth_percent),
            version=self.results_version,
            needles=[self.needle],
        )

        # Go generate the required length context and place your needle statement in
        context = await self.generate_context(context_length, depth_percent, evaluation_context)

        # Prepare your message to send to the model you're going to evaluate
        prompt = self.model_to_test.generate_prompt(context, self.retrieval_question)

        test_start_time = time.time()

        # Go see if the model can answer the question to pull out your random fact
        response = await self.model_to_test.evaluate_model(prompt)

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        # Compare the reponse to the actual needle you placed
        score = self.evaluation_model.evaluate_response(response)

        evaluation_context.model_response = response
        evaluation_context.score = score
        evaluation_context.test_duration_seconds = test_elapsed_time
        evaluation_context.test_timestamp_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')

        results = evaluation_context.model_dump()

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print(f"-- Test Summary -- ")
            print(f"Duration: {test_elapsed_time:.1f} seconds")
            print(f"Context: {context_length} tokens")
            print(f"Depth: {depth_percent}%")
            print(f"Insertion points (tokens): {evaluation_context.insertion_points}")
            print(f"Score: {score}")
            print(f"Response: {response}\n")

        model_name_safe = self.model_name.replace('.', '_').replace('/', '_').replace('\\', '_')
        context_file_location = f'{model_name_safe}_len_{context_length}_depth_{int(depth_percent*100)}'

        if self.save_contexts:
            evaluation_context.file_name = context_file_location
            results['file_name'] = context_file_location

            # Save the context to file for retesting
            if not os.path.exists('contexts'):
                os.makedirs('contexts')

            with open(f'contexts/{context_file_location}_context.txt', 'w') as f:
                f.write(context)

        if self.save_results:
            # Save the context to file for retesting
            if not os.path.exists('results'):
                os.makedirs('results')

            # Save the result to file for retesting
            with open(f'results/{context_file_location}_results.json', 'w') as f:
                json.dump(results, f)

        if self.seconds_to_sleep_between_completions:
            await asyncio.sleep(self.seconds_to_sleep_between_completions)

    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = 'results/'
        if not os.path.exists(results_dir):
            print("Results dir doesn't exist yet.")
            return False

        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    async def generate_context(self, context_length, depth_percent, evaluation_context=None):
        # Load up tiktoken so we navigate tokens more easily

        # Get your haystack dir files loaded into a string
        context = self.read_context_files()

        # Truncate the haystack dir essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context, insertion_point = self.insert_needle(context, depth_percent, context_length)

        if evaluation_context is not None:
            evaluation_context.insertion_points = [insertion_point]

        return context

    @staticmethod
    def _tokens_to_ids(tokens: list[TokenTextPair]) -> list[int]:
        return [token_id for _, token_id in tokens]

    def _has_sentence_ending(self, token_text: str) -> bool:
        stripped = token_text.strip()
        for ending in self.SENTENCE_ENDINGS:
            idx = stripped.find(ending)
            while idx != -1:
                following = stripped[idx + 1:]
                if not following or following.isspace() or all(
                    ch in self.SENTENCE_TRAILING_CHARS for ch in following
                ):
                    return True
                idx = stripped.find(ending, idx + 1)
        return False

    def _find_sentence_boundary(self, tokens: list[TokenTextPair], start_index: int) -> int:
        if not tokens:
            return 0

        start_index = max(0, min(start_index, len(tokens)))

        if start_index == 0:
            return 0

        for idx in range(start_index - 1, -1, -1):
            if self._has_sentence_ending(tokens[idx][0]):
                return idx + 1

        for idx in range(start_index, len(tokens)):
            if self._has_sentence_ending(tokens[idx][0]):
                return idx + 1

        return start_index


    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.model_to_test.encode_text_to_tokens(self.needle)
        tokens_context = self.model_to_test.encode_text_to_tokens(context)

        context_length -= self.final_context_length_buffer

        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            insertion_point = len(tokens_context)
            tokens_new_context = tokens_context + tokens_needle
        else:
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            insertion_point = self._find_sentence_boundary(tokens_context, insertion_point)
            tokens_new_context = (
                tokens_context[:insertion_point]
                + tokens_needle
                + tokens_context[insertion_point:]
            )

        new_context = self.model_to_test.decode_tokens(
            self._tokens_to_ids(tokens_new_context)
        )
        return new_context, insertion_point

    def get_context_length_in_tokens(self, context):
        return len(self.model_to_test.encode_text_to_tokens(context))

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)
        base_dir = os.path.abspath(os.path.dirname(__file__))  # Package directory

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(os.path.join(base_dir, self.haystack_dir, "*.txt")):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def encode_and_trim(self, context, context_length):
        token_pairs = self.model_to_test.encode_text_to_tokens(context)
        if len(token_pairs) > context_length:
            context = self.model_to_test.decode_tokens(
                self._tokens_to_ids(token_pairs), context_length
            )
        return context


    def get_results(self):
        return self.testing_results

    def print_start_test_summary(self):
        print("\n")
        print("Starting Needle In A Haystack Testing...")
        print(
            f"- Model: {self.model_name}"
        )
        print(
            f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}"
        )
        print(
            f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%"
        )
        print(f"- Needle: {self.needle.strip()}")
        print("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        asyncio.run(self.run_test())
