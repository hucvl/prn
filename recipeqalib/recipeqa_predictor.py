import json
import logging
import string
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, ListField, MetadataField, IndexField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor
from allennlp.common.util import JsonDict, sanitize
from allennlp.common.file_utils import cached_path
import numpy as np
import random
@Predictor.register('prn4recipeqar')
class ProceduralReasoningNetworksforRecipeQAPredictor(Predictor):
    """
    Wrapper for the :class:`~custom.models.ProceduralReasoningNetworksforRecipeQA` model.
    """

    @overrides
    def predict_json(self, inputs: str,task, cuda_device: int = -1) -> JsonDict:
        """
        Expects JSON that looks like ``{"text": "..."}``.
        """
        dataset_path = cached_path(inputs)
        dataset_instances = open(dataset_path, "rb")
        dataset_instances = json.load(dataset_instances)["data"]
        for item in dataset_instances:
            context_text = " ".join(
                [self._dataset_reader.prepro(x["body"]) + " @@SEP@@ " for x in item["context"] if len(x["body"]) != 0])
            step_indices_list = [len(self._dataset_reader.prepro(x["body"])) for x in item["context"] if len(x["body"]) != 0]
            # TODO: This is an issue with AllenNLP. Fix it later.
            tokenized_context = self._dataset_reader._tokenizer.tokenize(context_text)
            if len(tokenized_context) > 4000:
                print(item["recipe_id"])
            recipe_id = item["recipe_id"]  # TODO: Temporary solution to ingredients needs to be fixed.
            answer_ind = item["answer"]
            ing_list = self._dataset_reader._ing_json[recipe_id]

            if len(ing_list) < 1:
                # exit(0)
                continue

            ing_list = [item.replace(" ", "_") for item in ing_list]  # convert multiword ingredients to uniword

            ing_text = " ".join(ing_list)
            ingredient_tokens = self._dataset_reader._tokenizer.tokenize(ing_text)
            question_images = []
            answer_images = []
            question_text = ""
            tokenized_question = []
            answer_tokens = []
            placeholder_ind = 0
            if item["task"] == "textual_cloze" and "textual_cloze" in task:
                question_cloze = " ".join(item["question"])
                question_cloze = self._dataset_reader.prepro(question_cloze)
                question_cloze = question_cloze
                answer_tokens = []

                # TODO: This is a bug. Update the dataset for empty choices.
                skip = False

                for i, choice in enumerate(item["choice_list"]):
                    if self._dataset_reader.self.prepro(choice) == "":
                        skip = True
                        break

                if skip:
                    continue

                for choice in item["choice_list"]:
                    choice = self.prepro(choice)
                    tokenized_choice = self._tokenizer.tokenize(choice)
                    answer_tokens.append(tokenized_choice)

                placeholder_ind = question_cloze.strip().split().index("@placeholder")
                question_text = question_cloze.strip()
                tokenized_question = self._tokenizer.tokenize(question_text)

            elif "visual_cloze" in task and item["task"] == "visual_cloze":

                question_images = [[], [], [], []]
                for i, image in enumerate(item["question"]):
                    if image != "@placeholder":
                        ans = self._dataset_reader._lookup_array[image]
                        question_images[i] = ans

                    else:
                        zeros = np.zeros((2048,))
                        question_images[i] = zeros
                for choice in item["choice_list"]:
                    ans = self._dataset_reader._lookup_array[choice]
                    answer_images.append(ans)

            elif item["task"] == "visual_coherence" and "visual_coherence" in task:

                for i, image in enumerate(item["question"]):
                    ans = self._dataset_reader._lookup_array[image]
                    question_images.append(ans)

                for choice in item["choice_list"]:
                    ans = self._dataset_reader._lookup_array[choice]
                    answer_images.append(ans)

            elif item["task"] == "visual_ordering" and "visual_ordering" in task and len(task) != 1:
                answer_images = [[], [], [], []]
                for j, choices in enumerate(item["choice_list"]):
                    for k, choice in enumerate(choices):
                        ans = self._dataset_reader._lookup_array[choice]
                        answer_images[j].append(ans)
                x = random.randint(0, 3)
                question_images = answer_images[x]
            elif item["task"] == "visual_ordering" and "visual_ordering" in task and len(task) == 1:
                answer_images = [[], [], [], []]
                for j, choices in enumerate(item["choice_list"]):
                    for k, choice in enumerate(choices):
                        ans = self._dataset_reader._lookup_array[choice]
                        answer_images[j].append(ans)
                for i in range(4):
                    question_images = answer_images[i]
                    params = {}
                    params["context_text"] = context_text
                    params["answer_ind"] = answer_ind
                    params["step_indices_list"] = step_indices_list
                    params["ingredient_tokens"] = ingredient_tokens
                    params["question_text"] = question_text if item["task"] == "textual_cloze" else None
                    params["answer_tokens"] = answer_tokens if item["task"] == "textual_cloze" else None
                    params["tokenized_question"] = tokenized_question if item["task"] == "textual_cloze" else None
                    params["placeholder_ind"] = placeholder_ind if item["task"] == "textual_cloze" else None
                    params["question_images"] = question_images if item["task"] != "textual_cloze" else None
                    params["answer_images"] = answer_images if item["task"] != "textual_cloze" else None
                    params["recipe_id"] = recipe_id
                    params["task"] = item["task"]
                    params["qid"] = item["qid"]
                    instance = self._dataset_reader.text_to_instance(params["question_text"],
                                                     params["context_text"],
                                                     params["answer_ind"],
                                                     params["answer_tokens"],
                                                     params["tokenized_question"],
                                                     params["question_images"],
                                                     params["answer_images"],
                                                     params["step_indices_list"],
                                                     params["ingredient_tokens"],
                                                     params["placeholder_ind"],
                                                     params["recipe_id"],
                                                     params["task"],
                                                     params["qid"]
                                                     )
                    outputs = self._model.forward_on_instance(instance=instance)
                    sanitize(outputs)

            if item["task"] in task and task != ["visual_ordering"]:
                params = {}
                params["context_text"] = context_text
                params["answer_ind"] = answer_ind
                params["step_indices_list"] = step_indices_list
                params["ingredient_tokens"] = ingredient_tokens
                params["question_text"] = question_text if item["task"] == "textual_cloze" else None
                params["answer_tokens"] = answer_tokens if item["task"] == "textual_cloze" else None
                params["tokenized_question"] = tokenized_question if item["task"] == "textual_cloze" else None
                params["placeholder_ind"] = placeholder_ind if item["task"] == "textual_cloze" else None
                params["question_images"] = question_images if item["task"] != "textual_cloze" else None
                params["answer_images"] = answer_images if item["task"] != "textual_cloze" else None
                params["recipe_id"] = recipe_id
                params["task"] = item["task"]
                params["qid"] = item["qid"]
                instance = self._dataset_reader.text_to_instance(params["question_text"],
                                                 params["context_text"],
                                                 params["answer_ind"],
                                                 params["answer_tokens"],
                                                 params["tokenized_question"],
                                                 params["question_images"],
                                                 params["answer_images"],
                                                 params["step_indices_list"],
                                                 params["ingredient_tokens"],
                                                 params["placeholder_ind"],
                                                 params["recipe_id"],
                                                 params["task"],
                                                 params["qid"]
                                                 )
                outputs = self._model.forward_on_instance(instance=instance)
                sanitize(outputs)
        return outputs
