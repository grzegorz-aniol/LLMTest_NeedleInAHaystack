import asyncio
import json
import os
import time
from asyncio import Semaphore
from datetime import datetime, timezone

import numpy as np

from .evaluators import Evaluator
from .evaluation_context import EvaluationContext
from .llm_needle_haystack_tester import LLMNeedleHaystackTester
from .providers import ModelProvider


class LLMMultiNeedleHaystackTester(LLMNeedleHaystackTester):
    """
    Extends LLMNeedleHaystackTester to support testing with multiple needles in the haystack.
    
    Attributes:
        needles (list): A list of needles (facts) to insert into the haystack (context).
        model_to_test (ModelProvider): The model being tested.
        evaluator (Evaluator): The evaluator used to assess the model's performance.
        print_ongoing_status (bool): Flag to print ongoing status messages.
        eval_set (str): The evaluation set identifier.
    """
    def __init__(self, *args, 
                 needles=[], 
                 model_to_test: ModelProvider = None,
                 evaluator: Evaluator = None, 
                 print_ongoing_status = True,
                 eval_set = "multi-needle-eval-sf",
                 **kwargs):

        super().__init__(*args, model_to_test=model_to_test, **kwargs)
        self.needles = needles
        self.evaluator = evaluator
        self.model_to_test = model_to_test
        self.eval_set = eval_set
        self.model_name = self.model_to_test.model_name
        self.print_ongoing_status = print_ongoing_status
        self.insertion_percentages = []

    async def insert_needles(self, context, depth_percent, context_length):
        """Insert multiple needles into the context and return insertion points.

        Args:
            context (str): The original context string.
            depth_percent (float): The starting depth percent at which to insert the first needle.
            context_length (int): The total length of the context in tokens, adjusted for final buffer.

        Returns:
            tuple[str, list[int]]: The new context with needles inserted and the token indices
            at which each needle was inserted.
        """
        tokens_context = self.model_to_test.encode_text_to_tokens(context)
        context_length -= self.final_context_length_buffer

        # Calculate the total length of all needles in tokens
        total_needles_length = sum(len(self.model_to_test.encode_text_to_tokens(needle)) for needle in self.needles)

        # Ensure context length accounts for needles
        if len(tokens_context) + total_needles_length > context_length:
            tokens_context = tokens_context[:context_length - total_needles_length]
        
        # To evenly distribute the needles, we calculate the intervals they need to be inserted.
        depth_percent_interval = (100 - depth_percent) / len(self.needles)
        
        # Reset the insertion percentages list for the current context
        self.insertion_percentages = []
        insertion_points: list[int] = []

        # Insert needles at calculated points
        for needle in self.needles:

            tokens_needle = self.model_to_test.encode_text_to_tokens(needle)

            if depth_percent == 100:
                insertion_point = len(tokens_context)
                tokens_context = tokens_context + tokens_needle
            else:
                insertion_point = int(len(tokens_context) * (depth_percent / 100))
                insertion_point = self._find_sentence_boundary(tokens_context, insertion_point)
                tokens_context = (
                    tokens_context[:insertion_point]
                    + tokens_needle
                    + tokens_context[insertion_point:]
                )

            insertion_points.append(insertion_point)

            insertion_percentage = (insertion_point / len(tokens_context)) * 100
            self.insertion_percentages.append(insertion_percentage)

            depth_percent += depth_percent_interval

        token_ids = self._tokens_to_ids(tokens_context)
        new_context = self.model_to_test.decode_tokens(token_ids)
        return new_context, insertion_points

    def encode_and_trim(self, context, context_length):
        """
        Encodes the context to tokens and trims it to the specified length.
        
        Args:
            context (str): The context to encode and trim.
            context_length (int): The desired length of the context in tokens.
        
        Returns:
            str: The encoded and trimmed context.
        """
        tokens = self.model_to_test.encode_text_to_tokens(context)
        if len(tokens) > context_length:
            context = self.model_to_test.decode_tokens(
                self._tokens_to_ids(tokens), context_length
            )
        return context

    async def generate_context(self, context_length, depth_percent, evaluation_context: EvaluationContext | None = None):
        """Generate a context and insert all needles, recording insertion points.
        
        Args:
            context_length (int): The total length of the context in tokens.
            depth_percent (float): The depth percent for needle insertion.
            evaluation_context (EvaluationContext | None): Optional mutable evaluation
                context object to be populated with insertion points.
        
        Returns:
            str: The context with needles inserted.
        """
        context = self.read_context_files()
        context = self.encode_and_trim(context, context_length)
        context, insertion_points = await self.insert_needles(context, depth_percent, context_length)

        if evaluation_context is not None:
            evaluation_context.insertion_points = insertion_points

        return context
    
    async def evaluate_and_log(self, context_length, depth_percent):
        """Evaluate the model's performance with the generated context and log the results.
        
        Args:
            context_length (int): The length of the context in tokens.
            depth_percent (float): The depth percent for needle insertion.
        """
        if self.save_results and not self.overwrite_results:
            if self.result_exists(context_length, depth_percent):
                return

        # LangSmith
        ## TODO: Support for other evaluators 
        if self.evaluator.__class__.__name__ == "LangSmithEvaluator":  
            # Generate context without an EvaluationContext (LangSmith handles its own logging)
            context = await self.generate_context(context_length, depth_percent)

            test_start_time = time.time()
            print("EVALUATOR: LANGSMITH")
            chain = self.model_to_test.get_langchain_runnable(context)
            self.evaluator.evaluate_chain(
                chain,
                context_length,
                depth_percent,
                self.model_to_test.model_name,
                self.eval_set,
                len(self.needles),
                self.needles,
                self.insertion_percentages,
            )
            test_end_time = time.time()
            test_elapsed_time = test_end_time - test_start_time

        else:
            # Prepare evaluation context
            evaluation_context = EvaluationContext(
                model=self.model_to_test.model_name,
                context_length=int(context_length),
                depth_percent=float(depth_percent),
                version=self.results_version,
                needles=self.needles,
            )

            # Go generate the required length context and place your needle statements in
            context = await self.generate_context(context_length, depth_percent, evaluation_context)

            test_start_time = time.time()

            print("EVALUATOR: OpenAI Model")
            # Prepare your message to send to the model you're going to evaluate
            prompt = self.model_to_test.generate_prompt(context, self.retrieval_question)
            # Go see if the model can answer the question to pull out your random fact
            response = await self.model_to_test.evaluate_model(prompt)
            # Compare the reponse to the actual needle you placed
            score = self.evaluation_model.evaluate_response(response)

            test_end_time = time.time()
            test_elapsed_time = test_end_time - test_start_time

            evaluation_context.model_response = response
            evaluation_context.score = score
            evaluation_context.test_duration_seconds = test_elapsed_time
            evaluation_context.test_timestamp_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')

            results = evaluation_context.model_dump()

            self.testing_results.append(results)

            if self.print_ongoing_status:
                print (f"-- Test Summary -- ")
                print (f"Duration: {test_elapsed_time:.1f} seconds")
                print (f"Context: {context_length} tokens")
                print (f"Depth: {depth_percent}%")
                print (f"Insertion points (tokens): {evaluation_context.insertion_points}")
                print (f"Score: {score}")
                print (f"Response: {response}\n")

            context_file_location = f'{self.model_name.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'

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

    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needles: {self.needles}")
        print ("\n\n")

        if self.print_ongoing_status:
            self.print_start_test_summary()
        asyncio.run(self.run_test())
