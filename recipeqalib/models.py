import os
import logging
from typing import Any, Dict, List, Optional
from allennlp.data.fields import Field, TextField, ListField, MetadataField, IndexField, LabelField, ArrayField
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.onnx
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder, FeedForward
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1
from recipeqalib.relational_rnn_models import RelationalMemory

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("prn4recipeqa")
class ProceduralReasoningNetworksforRecipeQA(Model):
    """

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
    num_highway_layers : ``int``
        The number of highway layers to use in between embedding the input and passing it through
        the phrase layer.
    phrase_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and doing the bidirectional attention.
    similarity_function : ``SimilarityFunction``
        The similarity function that we will use when comparing encoded passage and question
        representations.
    modeling_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between the bidirectional
        attention and predicting span start and end.
    dropout : ``float``, optional (default=0.2)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    mask_lstms : ``bool``, optional (default=True)
        If ``False``, we will skip passing the mask to the LSTM layers.  This gives a ~2x speedup,
        with only a slight performance decrease, if any.  We haven't experimented much with this
        yet, but have confirmed that we still get very similar performance with much faster
        training times.  We still use the mask for all softmaxes, but avoid the shuffling that's
        required when using masking with pytorch LSTMs.
    memory_enabled :  optional (default=``False``)
        If True, the memory module is defined and will be used.
    memory_update : optional (default=``False``)
        If True, the memory module will be updated by encoded steps.
    memory_concat : optional (default=``False``)
        If True, the outputs of the memory operations will be concatted bidafwise(Like final merged passage).
    save_memory_snapshots: optional (default=``False``)
        If True, memory states will be saved.
    save_entity_embeddings: optional (default=``False``)
        If True, the entity embeddings in memory slots will be saved.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    answer_layer_image : ``Seq2SeqEncoder``, optional (default=``None``)
        If provided, will be used to encode the answers(as images) for calculating the similarity.
    answer_layer_text : ``Seq2SeqEncoder``, optional (default=``None``)
        If provided, will be used to encode the answers(as text) for calculating the similarity.
    question_image_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        If provided, will be used to encode the answers(as text) for calculating the similarity.
    question_image_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        If provided, will be used to encode the question images.
    step_layer : ``Seq2SeqEncoder``, optional (default=``None``)
        If provided, will be used to encode the steps.
    num_heads : ``int``, optional (default=``2``)
        If provided, will set the number of heads of multi-attention in memory unit.
    num_slots : ``int``, optional (default=``61``)
        If provided, will set the number of slots in memory unit.We calculated the max ingredients in a recipe is 61.
    last_layer_hidden_dims : ``List[int]``, optional (default=``None``)
        If provided, will be used as dimensions of last layer.
    last_layer_num_layers : ``int``, optional (default=``4``)
        If provided, will be used to set number of layers of last layer.
    projection_input_dim : ``int``, optional (default=``2048``)
        If provided, will be used as input dimension to project encoded images.
    projection_hidden_dims : ``List[int]``, optional (default=``None``)
        If provided, will be used as dimensions of projection layer.
    save_step_wise_attentions : ``bool``, optional (default=``false``)
        If provided, will save step wise attentions in memory unit.
    """
    #TODO : There are some variables that are not optional it should be fixed.
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 num_highway_layers: int,
                 phrase_layer: Seq2SeqEncoder,
                 similarity_function: SimilarityFunction,
                 modeling_layer: Seq2SeqEncoder,
                 modeling_layer_memory: Seq2SeqEncoder,
                 margin: float,
                 max: float,
                 dropout: float = 0.2,
                 mask_lstms: bool = False,
                 memory_enabled: bool = False,
                 memory_update: bool = True,
                 memory_concat: bool = False,
                 save_memory_snapshots: bool = False,
                 save_entity_embeddings: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 answer_layer_image: Seq2SeqEncoder = None,
                 answer_layer_text: Seq2SeqEncoder = None,
                 question_image_encoder: Seq2SeqEncoder = None,
                 step_layer: Seq2SeqEncoder = None,
                 num_heads: int = 2,
                 num_slots: int = 61, # Maximum number of entities in the training set.
                 last_layer_hidden_dims: List[int] = None,
                 last_layer_num_layers: int = 4,
                 projection_input_dim: int = 2048,
                 projection_hidden_dims: List[int] = None,
                 save_step_wise_attentions=False) -> None:

        super(ProceduralReasoningNetworksforRecipeQA, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._highway_layer = TimeDistributed(Highway(text_field_embedder.get_output_dim(),
                                                      num_highway_layers))
        self._phrase_layer = phrase_layer
        self._matrix_attention = LegacyMatrixAttention(similarity_function)
        self._modeling_layer = modeling_layer
        self._modeling_layer_memory = modeling_layer_memory
        self.margin = torch.FloatTensor([margin]).cuda()
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6).cuda()
        self.for_max = torch.FloatTensor([max]).cuda()
        self._memory_enabled = memory_enabled
        self._memory_update = memory_update
        self._memory_concat = memory_concat
        self._save_memory_snapshots = save_memory_snapshots
        self._save_entity_embeddings = save_entity_embeddings
        self._step_layer = step_layer
        self._label_acc = CategoricalAccuracy()
        self.save_step_wise_attentions = save_step_wise_attentions

        if self._memory_enabled:
            head_size = int(step_layer.get_output_dim() / num_heads)
            self.mem_module = RelationalMemory(mem_slots=num_slots,
                                               head_size=head_size,
                                               input_size=head_size * num_heads,
                                               num_heads=num_heads,
                                               num_blocks=1,
                                               forget_bias=1.,
                                               input_bias=0.,
                                               ).cuda(0)

            last_layer_input_dim = 10 * modeling_layer.get_output_dim()
        else:
            last_layer_input_dim = 5 * modeling_layer.get_output_dim()
        self._activation = torch.nn.Tanh()
        self._last_layer = FeedForward(last_layer_input_dim, last_layer_num_layers, last_layer_hidden_dims,
                                       self._activation, dropout)
        self._answer_layer_image = answer_layer_image  # uses image encoder for image input
        self._answer_layer_text = answer_layer_text  # uses text encoder for text input
        self._question_image_encoder = question_image_encoder  # converts question image inputs to encoding dim
        self._vocab = vocab
        # TODO: Replace hard coded parameters with config parameters
        self._mlp_projector = TimeDistributed(torch.nn.Sequential(
            torch.nn.Dropout(0.1, inplace=False),
            torch.nn.Linear(projection_input_dim, projection_hidden_dims[0]),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.1, inplace=False),
            torch.nn.Linear(projection_hidden_dims[0], projection_hidden_dims[1]),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.1, inplace=False),
            torch.nn.Linear(projection_hidden_dims[1], projection_hidden_dims[2]),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.1, inplace=False),
            torch.nn.Linear(projection_hidden_dims[2], projection_hidden_dims[3]),
        ))
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        self._mask_lstms = mask_lstms

        if self._save_memory_snapshots:
            if os.path.isfile('memory_snapshots_by_recipe.pkl'):  # make sure we start with a clean file
                os.remove('memory_snapshots_by_recipe.pkl')

        if self._save_entity_embeddings:
            if os.path.isfile('entity_embeddings_final.pkl'):  # make sure we start with a clean file
                os.remove('entity_embeddings_final.pkl')
        initializer(self)

    # TODO: answers should be a list not individual parameters!
    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                a1: Any,
                a2: Any,
                a3: Any,
                a4: Any,
                metadata: List[Dict[str, Any]],
                step_indice_list: Any = None,
                answer_indices: IndexField = None,
                ingredients: Dict[str, torch.LongTensor] = None,
                task:str = None) -> Dict[
        str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``.
        passage : Dict[str, torch.LongTensor]
            From a ``TextField``.  The model assumes that this passage contains the answer to the
            question, and predicts the beginning and ending positions of the answer within the
            passage.
        metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question ID, original passage text, and token
            offsets into the passage for each instance in the batch.  We use this for computing
            official metrics using the official SQuAD evaluation script.  The length of this list
            should be the batch size, and each dictionary should have the keys ``id``,
            ``original_passage``, and ``token_offsets``.  If you only want the best span string and
            don't care about official mtrics, you can omit the ``id`` key.
        step_indice_list : ``Any``, Optional
            If the memory is activated we update the memory step by step.
            To get the beginning and the end of a step we use this list.
            It contains end index of every step in the corresponding recipe.
        answer_indices : ``Any``, Optional
            We calculate the ranking loss using these indices as ground truth indices.
        ingredients: Dict[str, torch.LongTensor] , Optional
            If the memory is activated the memory is initialized with the ingredients of a recipe.

        Returns
        -------
        An output dictionary consisting of:
        loss : torch.FloatTensor
            A scalar loss to be optimised.
        """
        self._task = metadata[0]["task"]  # TODO: Make it instance dependant

        if self._task in ["visual_cloze", "visual_ordering", "visual_coherence"]:
            embedded_question = self._highway_layer(self._question_image_encoder(question, None))
            question_mask = None
        else:
            embedded_question = None
            question_mask = None

        embedded_passage = self._highway_layer(self._text_field_embedder(passage))
        batch_size = embedded_question.size(0)
        passage_length = embedded_passage.size(1)
        passage_mask = util.get_text_field_mask(passage).float()
        memory_mask = util.get_text_field_mask(ingredients).float()
        passage_lstm_mask = passage_mask if self._mask_lstms else None
        question_lstm_mask = question_mask if self._mask_lstms else None
        # memory_lstm_mask = memory_mask
        # encoded_all_steps = torch.FloatTensor(size=[batch_size, passage_length, 1224]).cuda(dvc)

        if self._memory_enabled:
            memory_length = self.mem_module.mem_slots
            dvc = memory_mask.get_device()
            difference = memory_length - memory_mask.size(1)
            pad = torch.zeros((batch_size, difference)).cuda(dvc).float()
            memory_mask = torch.cat([memory_mask, pad], 1).cuda(dvc).float()
            if self._save_memory_snapshots:
                if os.path.isfile('memory_snapshots_by_recipe.pkl'):
                    with open('memory_snapshots_by_recipe.pkl', 'rb') as handle:
                        memory_snapshots_by_recipe = pickle.load(handle)
                else:
                    memory_snapshots_by_recipe = {}

            if self._save_entity_embeddings:
                if os.path.isfile('entity_embeddings_final.pkl'):
                    with open('entity_embeddings_final.pkl', 'rb') as handle:
                        entity_embeddings = pickle.load(handle)
                else:
                    entity_embeddings = {}

            # shape = (batch_size,ingredient_length,embedding_dim)
            ingredient_embeddings_batch = self._text_field_embedder(ingredients)

            last_mems = []
            # first encode each steps, update the memory according to that step.
            for i in range(batch_size):
                memory_snapshots = []
                ingredient_embeddings = ingredient_embeddings_batch[i]
                memory = self.mem_module.initial_state(
                    ingredient_embeddings=ingredient_embeddings.unsqueeze(0),
                    memory_mask=memory_mask[i])
                memory_snapshots.append(memory)  # save initial memory as a snapshot

                if self._memory_update:
                    step_indices = list(
                        map(int,
                            list(step_indice_list[i].cpu().numpy())))
                    # skip paddings
                    if step_indices[-1] == 0:
                        step_indices = step_indices[0:step_indices.index(0)]

                    last_ind = 0

                    for j, ind in enumerate(step_indices):  # iterate each step in the context

                        embedded_step = embedded_passage[i, last_ind: ind + 1, :]
                        encoded_step = self._step_layer(embedded_step.unsqueeze(0), None)[:, -1, :]
                        last_ind = ind + 1
                        # Updating memory
                        if self.save_step_wise_attentions:
                            weights_list, logit, memory = self.mem_module(encoded_step.squeeze(0), memory, targets=None,
                                                                          treat_input_as_matrix=False,
                                                                          memory_mask=memory_mask[i])
                        else:

                            logit, memory = self.mem_module(encoded_step.squeeze(0), memory, targets=None,
                                                            treat_input_as_matrix=False, memory_mask=memory_mask[i])
                        memory_snapshots.append(memory)

                        detached_memory = memory.squeeze(0).cpu().detach().numpy()

                        if self._save_entity_embeddings:
                            for k, ing in enumerate(metadata[i]["ingredients"]):
                                key = str(metadata[i]["recipe_id"]) + ":" + str(ing) + ":" + str(j)
                                value = detached_memory[k]
                                entity_embeddings[key] = value

                last_mems.append(memory_snapshots[-1])

                if self._save_memory_snapshots:
                    memory_snapshots_by_recipe[metadata[i]["recipe_id"]] = [
                        [memory_snapshots[0], memory_snapshots[-1], len(step_indices)]]

            encoded_memory = torch.cat(last_mems, dim=0)  # we take the last memory snapshot as the memory embedding

            if self._save_entity_embeddings:
                with open('entity_embeddings_final.pkl', 'wb') as handle:  # save entity embeddings to a file
                    pickle.dump(entity_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if self._save_memory_snapshots:
                with open('memory_snapshots_by_recipe.pkl', 'wb') as handle:  # save memory snaptshots to a file
                    pickle.dump(memory_snapshots_by_recipe, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # RM LAYER ENDS

        encoded_passage = self._dropout(self._phrase_layer(embedded_passage, passage_lstm_mask))
        encoded_question = self._dropout(self._phrase_layer(embedded_question, question_lstm_mask))
        encoding_dim = encoded_question.size(-1)
        batch_size = passage_mask.size(0)

        # ATTENTION LAYER STARTS
        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)

        if self._memory_enabled:
            memory_question_similarity = self._matrix_attention(encoded_memory, encoded_question)

        passage_question_attention = F.softmax(passage_question_similarity, dim=-1)
        question_passage_similarity = passage_question_similarity.max(dim=-1)[0].squeeze(-1)

        if self._memory_enabled:
            memory_question_attention = F.softmax(memory_question_similarity, dim=-1)
            question_memory_similarity = memory_question_similarity.max(dim=-1)[0].squeeze(-1)

            if len(question_memory_similarity.size()) == 1:
                question_memory_similarity = question_memory_similarity.unsqueeze(-1)

        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)
        # Shape: (batch_size, passage_length)
        question_passage_attention = util.masked_softmax(question_passage_similarity, passage_mask)
        # Shape: (batch_size, encoding_dim)
        question_passage_vector = util.weighted_sum(encoded_passage, question_passage_attention)

        # Shape: (batch_size, passage_length, encoding_dim)
        tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(batch_size,
                                                                                    passage_length,
                                                                                    encoding_dim)

        if self._memory_enabled:
            question_memory_attention = util.masked_softmax(question_memory_similarity, memory_mask)
            question_memory_vector = util.weighted_sum(encoded_memory, question_memory_attention)
            memory_question_vector = util.weighted_sum(encoded_question, memory_question_attention)
            tiled_question_memory_vector = question_memory_vector.unsqueeze(1).expand(batch_size,
                                                                                      memory_length,
                                                                                      encoding_dim)

        # ATTENTION LAYER ENDS

        # MODELING LAYER STARTS
        # We concatenate different vectors and use them for classification.
        # Shape: (batch_size, passage_length, encoding_dim * 4)
        final_merged_passage = torch.cat([encoded_passage,
                                          passage_question_vectors,
                                          encoded_passage * passage_question_vectors,
                                          encoded_passage * tiled_question_passage_vector],
                                         dim=-1)
        if self._memory_enabled:
            if self._memory_concat:
                final_merged_memory = torch.cat([encoded_memory,
                                                 memory_question_vector,
                                                 encoded_memory * memory_question_vector,
                                                 encoded_memory * tiled_question_memory_vector],
                                                dim=-1)
            else:
                final_merged_memory = encoded_memory

        modeled_passage = self._dropout(self._modeling_layer(final_merged_passage, passage_lstm_mask))[:, -1, :]
        modeled_passage = torch.cat([final_merged_passage[:, -1, :], modeled_passage], dim=-1)

        if self._memory_enabled:
            modeled_memory = self._dropout(self._modeling_layer_memory(final_merged_memory, memory_mask))[:, -1, :]
            modeled_memory = torch.cat([final_merged_memory[:, -1, :], modeled_memory], dim=-1)

        # MODELING LAYER ENDS
        # OUTPUT LAYER STARTS
        if self._memory_enabled:
            last_layer_input = self._dropout(torch.cat([modeled_passage, modeled_memory], dim=-1))
        else:
            last_layer_input = self._dropout(modeled_passage)

        last_layer_logits = self._last_layer(last_layer_input)

        output_dict = {
            "question_memory_attentions": {},
            "memory_question_attentions": {},
            "step_wise_attentions": [],
            "ingredients": []
        }
        loss = Variable(torch.FloatTensor([1]), requires_grad=True).cuda()
        dvc = loss.get_device()
        model_answers = np.ones((batch_size, 1))
        embedded_answers = []

        # First we take the embedding of the words/images than we feed them to an rnn
        # we take the last hidden state from the rnn to calculate cosine distance.
        if self._task in ["visual_coherence", "visual_cloze"]:
            # for visual tasks we encode images with image encoder rnn.
            embedded_answers.append(
                self._mlp_projector(a1.unsqueeze(0)).squeeze(0))
            embedded_answers.append(
                self._mlp_projector(a2.unsqueeze(0)).squeeze(0))
            embedded_answers.append(
                self._mlp_projector(a3.unsqueeze(0)).squeeze(0))
            embedded_answers.append(
                self._mlp_projector(a4.unsqueeze(0)).squeeze(0))
        elif self._task == "visual_ordering":
            projected_a1 = self._mlp_projector(a1)
            projected_a2 = self._mlp_projector(a2)
            projected_a3 = self._mlp_projector(a3)
            projected_a4 = self._mlp_projector(a4)
            # There are multiple images in choices.So we have to consider it differently from other visual tasks.

            embedded_answers.append(torch.sum(self._dropout((self._answer_layer_image(projected_a1,
                                                                            None))),dim=1))
            embedded_answers.append(torch.sum(self._dropout((self._answer_layer_image(projected_a2,
                                                                                      None))),dim=1))
            embedded_answers.append(torch.sum(self._dropout((self._answer_layer_image(projected_a3,
                                                                                      None))),dim=1))
            embedded_answers.append(torch.sum(self._dropout((self._answer_layer_image(projected_a4,
                                                                                      None))),dim=1))
        else:
            embedded_answers = None

        # in this loop for every instance we are taking cosine similarities for ranking loss and classification.
        for i in range(batch_size):
            if self._memory_enabled and self.save_step_wise_attentions:
                output_dict["question_memory_attentions"] = question_memory_attention
                output_dict["memory_question_attentions"] = memory_question_attention
                output_dict["step_wise_attentions"].append(weights_list)
                output_dict["ingredients"].append(metadata[i]["ingredients"])
            ground_truth_ind = answer_indices[i].item()
            sim_right = self.cos(last_layer_logits[i], embedded_answers[ground_truth_ind][i]).cuda(dvc)
            sims = [[], [], [], []]
            sims[ground_truth_ind] = sim_right

            for j in range(4):  # for multiple choice we have 4 answers in our choice_list

                if j == ground_truth_ind:
                    continue
                sim_wrong = self.cos(last_layer_logits[i], embedded_answers[j][i]).cuda(dvc)
                sims[j] = sim_wrong
                loss += torch.max(self.for_max.cuda(dvc), self.margin.cuda(dvc) - sim_right + sim_wrong)

            ans = sims.index(max(sims))
            model_answers[i] = ans

        model_answers = torch.from_numpy(model_answers)
        loss = loss / batch_size
        model_answers = self.get_one_hot(model_answers.squeeze(-1).long(), 4)
        output_dict["loss"] = loss[None]
        acc = [self._label_acc.get_metric(False)]
        output_dict["acc"] = acc
        answer_indices = answer_indices.squeeze(-1)
        self._label_acc(model_answers, answer_indices)
        # OUTPUT LAYER ENDS
        return output_dict

    # Calculate accuracy
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "acc": self._label_acc.get_metric(reset)
        }

    # Turn an index vector to one hot vector for calculating accuracy
    def get_one_hot(self, labels, num_classes):
        y = torch.eye(num_classes)
        return y[labels]
