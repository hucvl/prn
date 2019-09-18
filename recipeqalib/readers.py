import json
import logging
import string
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, OrderedDict
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, ListField, MetadataField, IndexField, LabelField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
import numpy as np
import random
from subprocess import Popen, PIPE
from io import BytesIO
import pickle
import re
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("recipeqa")
class RecipeQAReader(DatasetReader):
    """
    Reads a JSON-formatted RecipeQA file and returns a ``Dataset`` where the ``Instances`` have four
    fields: ``question``, a ``TextField``, ``passage``, another ``TextField`` ,
    both ``IndexFields`` into the ``passage`` ``TextField``.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    task : str, must
        We use this to determine which task the model should perform and for which task we should create instances
    lookup_path: str, optional
        We use this to get feature vectors from lookup array.Each key,value pair consists of image_name: image index in lookup array
    feat_path:str, optional
        We use this to get lookup array of feature vectors of images.
    ing_path:str, optional
        We use this to get ingredient file.

    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False, task: str = None, lookup_path: str = None, feat_path: str = None,ing_path : str = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._task = task or "what"
        if lookup_path: # Not being used currently.
            pass
            # key,value pairs consists of image_name image_ind in lookup array
            # file = open(lookup_path)
            # self._lookup_json = json.load(file)
        if feat_path:
            # saves image features row by row
            im_feat_path = cached_path(feat_path)
            imfeats = open(im_feat_path, "rb")
            self._lookup_array = pickle.load(imfeats)
        if ing_path:
            ing_file = open("./data/" + ing_path)
            self._ing_json = json.load(ing_file)

        # ingredient_counts = self.load_json('./data/recipeqa_ingredient_histogram_final.json')
        # ingredients_dict = {}
        #
        # for k, v in ingredient_counts.items():
        #     k = k.lower()
        #     if k not in ingredients_dict:
        #         ingredients_dict[k] = k.replace(" ", "_")

        # self._ingredients_dict = ingredients_dict
        #
        # # find multi-word ingredients and their token counts
        # ingredient_tokens = dict()
        # for k, v in ingredients_dict.items():
        #     if k in ingredient_tokens:
        #         continue
        #     else:
        #         ingredient_tokens[k] = len(k.split())
        #
        # self._ingredients_with_5tokens = {k: ingredients_dict[k] for k, v in ingredient_tokens.items() if v == 5}
        # self._ingredients_with_4tokens = {k: ingredients_dict[k] for k, v in ingredient_tokens.items() if v == 4}
        # self._ingredients_with_3tokens = {k: ingredients_dict[k] for k, v in ingredient_tokens.items() if v == 3}
        # self._ingredients_with_2tokens = {k: ingredients_dict[k] for k, v in ingredient_tokens.items() if v == 2}
        # self._ingredients_with_1tokens = {k: ingredients_dict[k] for k, v in ingredient_tokens.items() if v == 1}

        self._image_dim = 2048

    def load_json(self, file_name):
        with open(file_name) as data_file:
            data = json.load(data_file, object_pairs_hook=OrderedDict)
        return data

    def white_space_fix(self, text):
        return ' '.join(text.split())

    def remove_punc(self, text):
        exclude = set(string.punctuation)
        exclude.add("@")
        return ''.join(ch for ch in text if ch not in exclude)

    def normalize(self, text):
        text = text.replace("''", '"').replace("``", '" ')
        return text

    def fix_punc(self, text):
        puncs = ".,;!?"

        for punc in puncs:
            text = text.replace(punc, " " + punc + "")

        return text

    def prepro(self, text):
        # The following replacements are suggested in the paper
        # BidAF (Seo et al., 2016)
        text = text.strip().replace("\n", " ")
        text = self.fix_punc(text)
        text = text.encode("ascii", errors="ignore").decode()
        text = ' '.join(text.split())
        text = text.strip()

        return text

    def substitute_ingredients(text, ingredients):

        for k, v in ingredients.items():
            pattern = re.compile(k.lower(), re.IGNORECASE)
            text = pattern.sub(v, text)

        return text

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        logger.info("Reading file at %s", file_path)

        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']

        logger.info("Reading the dataset")
        task_key = "_".join(self._task)
        logger.info("The task is " + task_key)

        dataset_instances = {task_key: [x for x in dataset if x["task"] in self._task]}

        for item in dataset_instances[task_key]:
            context_text = " ".join([self.prepro(x["body"]) + " @@SEP@@ " for x in item["context"] if len(x["body"]) != 0])
            step_indices_list = [len(self.prepro(x["body"])) for x in item["context"] if len(x["body"]) != 0]
            # TODO: This is due to hardware constraints we filtered out the passages larger than 4000 tokens.
            tokenized_context = self._tokenizer.tokenize(context_text)
            if len(tokenized_context) > 4000:
                continue
            recipe_id = item["recipe_id"]
            answer_ind = item["answer"]
            ing_list = self._ing_json[recipe_id]

            if len(ing_list) < 1: # Skip instances which has no entities.
                print(recipe_id, len(ing_list), " ZERO INGREDIENTS!")
                continue

            ing_list = [item.replace(" ", "_") for item in ing_list] # convert multiword ingredients to uniword

            ing_text = " ".join(ing_list)
            ingredient_tokens = self._tokenizer.tokenize(ing_text)
            question_images = []
            answer_images = []
            question_text = ""
            tokenized_question = []
            answer_tokens = []
            placeholder_ind = 0

            if item["task"] == "visual_cloze":

                question_images = [[],[],[],[]]

                for i, image in enumerate(item["question"]):
                    if image != "@placeholder":
                        ans = self._lookup_array[image]
                        question_images[i] = ans
                    else:
                        zeros = np.zeros((self._image_dim,))
                        question_images[i] = zeros

                for choice in item["choice_list"]:
                    ans = self._lookup_array[choice]
                    answer_images.append(ans)

            elif item["task"] == "visual_coherence":

                for i, image in enumerate(item["question"]):
                    ans = self._lookup_array[image]
                    question_images.append(ans)

                for choice in item["choice_list"]:
                    ans = self._lookup_array[choice]
                    answer_images.append(ans)

            elif item["task"] == "visual_ordering":

                answer_images = [[], [], [], []]

                for j, choices in enumerate(item["choice_list"]):
                    for k, choice in enumerate(choices):
                        ans = self._lookup_array[choice]
                        answer_images[j].append(ans)

                for i,images in enumerate(answer_images):
                    question_images = images
                    params  = {}
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
                    instance = self.text_to_instance(params["question_text"],
                                                     params["context_text"],
                                                     params["answer_ind"],
                                                     params["answer_tokens"],
                                                     params["tokenized_question"],
                                                     params["question_images"],
                                                     params["answer_images"],
                                                     params["step_indices_list"],
                                                     params["ingredient_tokens"],
                                                     None,
                                                     params["recipe_id"],
                                                     params["task"],
                                                     params["qid"]
                                                     )
                    yield instance
            if item["task"] != "visual_ordering":
                
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
                instance = self.text_to_instance(params["question_text"],
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
    
                yield instance

    @overrides
    def text_to_instance(self,
                         question_text: str = None,
                         passage_text: str = None,
                         answer_ind: int = None,
                         answer_tokens: List[Token] = None,
                         question_tokens: List[Token] = None,
                         question_images: Any = None,
                         answer_images: Any = None,
                         step_indice_list: Any = None,
                         ingredient_tokens: Any = None,
                         placeholder_ind: int = None,
                         recipe_id:str = None,
                         task:str = None,
                         qid:str = None) -> Instance:

        # pylint: disable=arguments-differ
        if not question_tokens and question_text:
            question_tokens = self._tokenizer.tokenize(question_text)
        passage_tokens = self._tokenizer.tokenize(passage_text)
        total = 0
        if question_images:
            question_images = np.array(question_images)

        def make_multi_modal_instance(passage_tokens: List[Token],
                                      token_indexers: Dict[str, TokenIndexer],
                                      answer_images: List[Any] = None,
                                      question_images: Any = None,
                                      step_indice_list: Any = None):

            fields: Dict[str, Field] = {}

            # This is separate so we can reference it later with a known type.
            fields['passage'] = TextField(passage_tokens, token_indexers)

            if question_images.all() is not None:
                fields['question'] = ArrayField(question_images)

            fields['a1'] = ArrayField(np.array(answer_images[0]))
            fields['a2'] = ArrayField(np.array(answer_images[1]))
            fields['a3'] = ArrayField(np.array(answer_images[2]))
            fields['a4'] = ArrayField(np.array(answer_images[3]))
            fields["task"] = MetadataField(task)
            step_indice_list = [i for i, x in enumerate(passage_tokens) if str(x) == "@@SEP@@"]
            fields["step_indice_list"] = ArrayField(np.array(step_indice_list, dtype="int32"))
            fields["ingredients"] = TextField(ingredient_tokens, token_indexers)

            metadata = {
                "ingredients": ingredient_tokens,
                'passage_tokens': [token.text for token in passage_tokens],
                'task': task,
                'recipe_id': recipe_id,
                'qid': qid
            }

            fields["answer_indices"] = IndexField(answer_ind, passage_tokens)
            fields['metadata'] = MetadataField(metadata)

            return Instance(fields)

        def make_reading_comprehension_instance(question_tokens: List[Token],
                                                passage_tokens: List[Token],
                                                token_indexers: Dict[str, TokenIndexer],
                                                answer_ind: int,
                                                answer_tokens: List[Token] = None,
                                                additional_metadata: Dict[str, Any] = None,
                                                ingredient_tokens:Any = None) -> Instance:


            additional_metadata = additional_metadata or {}
            fields: Dict[str, Field] = {}
            question_offsets = [(token.idx, token.idx + len(token.text)) for token in question_tokens]
            question_field = TextField(question_tokens, token_indexers)
            step_indice_list = [i for i, x in enumerate(passage_tokens) if str(x) == "@@SEP@@"]
            fields['passage'] = TextField(passage_tokens, token_indexers)
            fields['question'] = question_field
            answer_list = []
            fields["ingredients"] = TextField(ingredient_tokens,token_indexers)

            for answers in answer_tokens:
                answer_list.append(TextField(answers, token_indexers))

            fields['a1'] = answer_list[0]
            fields['a2'] = answer_list[1]
            fields['a3'] = answer_list[2]
            fields['a4'] = answer_list[3]
            fields["step_indice_list"] = ArrayField(np.array(step_indice_list, dtype="int32"))

            metadata = {'original_question': question_text, 'token_offsets': question_offsets,
                        'question_tokens': [token.text for token in question_tokens],
                        "task": task,
                        'passage_tokens': [token.text for token in passage_tokens],
                        "placeholder_ind": placeholder_ind,
                        "answers": answer_tokens,
                        "recipe_id": recipe_id,
                        "qid": qid
                    }

            fields["answer_indices"] = IndexField(answer_ind, question_tokens)

            metadata.update(additional_metadata)
            fields['metadata'] = MetadataField(metadata)

            return Instance(fields)

        if task in  ["visual_cloze","visual_coherence","visual_ordering"]:
            return make_multi_modal_instance(passage_tokens,
                                             self._token_indexers,
                                             answer_images,
                                             question_images,
                                             step_indice_list=step_indice_list)
