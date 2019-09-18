import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

from allennlp.commands import main
from recipeqalib.models import ProceduralReasoningNetworksforRecipeQA
from recipeqalib.readers import RecipeQAReader
from recipeqalib.recipeqa_predictor import ProceduralReasoningNetworksforRecipeQAPredictor

if __name__ == "__main__":
    main(prog="python -m allennlp.run")
