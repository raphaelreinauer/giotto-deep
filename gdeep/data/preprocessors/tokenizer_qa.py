from collections import Counter
from typing import Callable, Tuple, List, Optional, Sequence

import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer  # type: ignore
from gdeep.utility import DEVICE

from ..abstract_preprocessing import AbstractPreprocessing
# type definition
Tensor = torch.Tensor



class TokenizerQA(AbstractPreprocessing[Tuple[str,str,List[str],List[int]],
                                        Tuple[Tensor, Tensor]]):
    """Class to preprocess text dataloaders for Q&A
    tasks. The type of dataset is assumed to be of the
    form ``(string,string,list[string], list[string])``.

    Args:
        vocabulary:
            the torch vocabulary
        tokenizer :
            the tokenizer of the source text

    Examples::

        from gdeep.data import TorchDataLoader
        from gdeep.data import TransformingDataset
        from gdeep.data.preprocessors import TokenizerQA

        dl = TorchDataLoader(name="SQuAD2", convert_to_map_dataset=True)
        dl_tr, dl_ts = dl.build_dataloaders()

        textds = TransformingDataset(dl_tr_str.dataset,
                               TokenizerQA())

    """
    is_fitted: bool
    max_length: int
    vocabulary: Optional[Sequence[str]]
    tokenizer: Optional[Callable[[str], List[str]]]
    counter: "Counter[List[str]]"

    def __init__(self, vocabulary:Optional[Sequence[str]]=None,
                 tokenizer:Optional[Callable[[str], List[str]]]=None):
        if tokenizer is None:
            self.tokenizer = get_tokenizer('basic_english')
        else:
            self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.max_length = 0
        self.is_fitted = False

    def fit_to_dataset(self, dataset: Dataset[Tuple[str,str,List[str],List[int]]]) -> None:
        """Method to fit the vocabulary to the input text"""
        counter = Counter()  # for the text
        for (context, question, answer, init_position) in dataset:  # type: ignore
            counter.update(self.tokenizer(context))
            self.max_length = max(self.max_length, len(self.tokenizer(context)))
            counter.update(self.tokenizer(question))
            self.max_length = max(self.max_length, len(self.tokenizer(question)))

        if self.vocabulary is None:
            self.vocabulary = Vocab(counter)
        self.pad_item = self.vocabulary["."]
        self.is_fitted = True

    def __call__(self, datum: Tuple[str, str, List[str], List[int]]) -> Tuple[Tensor, Tensor]:
        """This method implement the transformation once fitted."""
        if not self.is_fitted:
            self.load_pretrained(".")
        text_pipeline = lambda x: [self.vocabulary[token] for token in # type: ignore
                                   self.tokenizer(x)]  # type: ignore

        processed_context = torch.tensor(text_pipeline(datum[0]),
                                      dtype=torch.int64).to(DEVICE)
        out_context = torch.cat([processed_context,
                         self.pad_item * torch.ones(self.max_length - processed_context.shape[0]
                                               ).to(DEVICE)])
        processed_question = torch.tensor(text_pipeline(datum[1]),
                                         dtype=torch.int64).to(DEVICE)

        out_question = torch.cat([processed_question,
                         self.pad_item * torch.ones(self.max_length - processed_question.shape[0]
                                               ).to(DEVICE)])

        pos_init_char = datum[3][0]
        pos_init = len(self.tokenizer(datum[0][:pos_init_char]))
        pos_end = pos_init + len(self.tokenizer(datum[2][0]))

        return (torch.stack((out_context, out_question)).to(torch.long),
         torch.stack((torch.tensor(pos_init), torch.tensor(pos_end))).to(torch.long))


