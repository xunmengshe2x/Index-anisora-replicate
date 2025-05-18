# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for Idefics3.
"""

from typing import TYPE_CHECKING, List, Optional, Union

from transformers import BatchFeature
from transformers.image_utils import ImageInput, is_valid_image, load_image
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import AddedToken, BatchEncoding, PaddingStrategy, TextInput, TruncationStrategy
from transformers.utils import TensorType, logging
from .image_processing_idefics3 import Idefics3ImageProcessor
import transformers
transformers.Idefics3ImageProcessor = Idefics3ImageProcessor

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTokenizedInput

logger = logging.get_logger(__name__)


def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


def _prompt_split_image(image_seq_len, image_rows, image_cols, fake_token_around_image, image_token):
    """Prompt with expanded image tokens for when the image is split into patches."""
    text_split_images = ""
    for n_h in range(image_rows):
        for n_w in range(image_cols):
            text_split_images += (
                f"{fake_token_around_image}"
                + f"<row_{n_h + 1}_col_{n_w + 1}>"
                + f"{image_token}" * image_seq_len
            )
        text_split_images += "\n"

    text_split_images += (
        f"\n{fake_token_around_image}"
        + "<global-img>"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )
    return text_split_images


def _prompt_single_image(image_seq_len, fake_token_around_image, image_token):
    """Prompt with expanded image tokens for a single image."""
    return f"{fake_token_around_image}" + "<global-img>" + f"{image_token}" * image_seq_len + f"{fake_token_around_image}"


def get_image_prompt_string(image_rows, image_cols, image_seq_len, fake_token_around_image, image_token):
    if image_rows == 0 and image_cols == 0:
        return _prompt_single_image(
            image_seq_len, fake_token_around_image=fake_token_around_image, image_token=image_token
        )
    return _prompt_split_image(image_seq_len, image_rows, image_cols, fake_token_around_image, image_token)


class Idefics3Processor(ProcessorMixin):
    r"""
    Constructs a Idefics3 processor which wraps a LLama tokenizer and Idefics3 image processor into a single processor.

    [`Idefics3Processor`] offers all the functionalities of [`Idefics3ImageProcessor`] and [`Idefics3TokenizerFast`]. See
    the docstring of [`~IdeficsProcessor.__call__`] and [`~IdeficsProcessor.decode`] for more information.

    Args:
        image_processor (`Idefics3ImageProcessor`):
            An instance of [`Idefics3ImageProcessor`]. The image processor is a required input.
        tokenizer (`PreTrainedTokenizerBase`, *optional*):
            An instance of [`PreTrainedTokenizerBase`]. This should correspond with the model's text model. The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Idefics3ImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer=None, chat_template: str = None, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        self.fake_image_token = AddedToken("<fake_token_around_image>", normalized=False, special=True)
        self.image_token = AddedToken("<image>", normalized=False, special=True)
        self.end_of_utterance_token = AddedToken("<end_of_utterance>", normalized=False, special=True)

        tokens_to_add = {
            "additional_special_tokens": [self.fake_image_token, self.image_token, self.end_of_utterance_token]
        }
        tokenizer.add_special_tokens(tokens_to_add)
        

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def _extract_images_from_prompts(self, prompts):
        prompt_images = []
        for prompt in prompts:
            images = []
            for elem in prompt:
                if is_valid_image(elem):
                    images.append(elem)
                elif is_url(elem):
                    images.append(load_image(elem))
            prompt_images.append(images)
        return prompt_images

    def __call__(
        self,
        text: Union[TextInput, "PreTokenizedInput", List[TextInput], List["PreTokenizedInput"]] = None,
        images: Union[ImageInput, List[ImageInput], List[List[ImageInput]]] = None,
        image_seq_len: int = 169,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        is_split_into_words: bool = False,
        add_special_tokens: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchEncoding:
        """
        Processes the input prompts and returns a BatchEncoding.

        Example:

        ```python
        >>> import requests
        >>> from transformers import Idefics3Processor
        >>> from transformers.image_utils import load_image

        >>> processor = Idefics3Processor.from_pretrained("HuggingFaceM4/idefics3-8b", image_seq_len=2)
        >>> processor.image_processor.do_image_splitting = False  # Force as False to simplify the example

        >>> url1 = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        >>> url2 = "https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg"

        >>> image1, image2 = load_image(url1), load_image(url2)
        >>> images = [[image1], [image2]]

        >>> text = [
        ...     "<image>In this image, we see",
        ...     "bla bla bla<image>",
        ... ]
        >>> outputs = processor(text=text, images=images, return_tensors="pt", padding=True)
        >>> input_ids = outputs.input_ids
        >>> input_tokens = processor.tokenizer.batch_decode(input_ids)
        >>> print(input_tokens)
        ['<s><fake_token_around_image><image><image><fake_token_around_image> In this image, we see', '<s> bla bla bla<fake_token_around_image><image><image><fake_token_around_image>']
        ```

        Args:
            text (`Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).

                Wherever an image token, `<image>` is encountered it is expanded to
                `<fake_token_around_image>` + `<image>` * `image_seq_len` * <fake_token_around_image>`.
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`, *optional*):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. If is of type `List[ImageInput]`, it's assumed that this is for a single prompt i.e. of batch size 1.
            image_seq_len (`int`, *optional*):
                The length of the image sequence. If not provided, the default value of 169 is used.
            padding (`Union[bool, str, PaddingStrategy]`, *optional*, defaults to `False`):
                Padding strategy applied to the input ids. See [`PreTrainedTokenizerFast.pad`] for more information.
            truncation (`Union[bool, str, TruncationStrategy]`, *optional*):
                Truncation strategy applied to the input ids. See [`PreTrainedTokenizerFast.truncate`] for more information.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding/truncation length. See
                [`PreTrainedTokenizerFast.__call__`] for more information.
            is_split_into_words (`bool`, *optional*, defaults to `False`):
                Whether the input text is split into words or not. If set to `True`, the tokenizer will skip the
                tokenization process and assume the input is already tokenized.
            add_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether to add special tokens or not. See [`PreTrainedTokenizerFast.__call__`] for more information.
            return_tensors (`Union[str, TensorType]`, *optional*):
                If set, will return tensors of a particular framework. See [`PreTrainedTokenizerFast.__call__`] for more
                information.
        """
        n_images_in_text = []
        n_images_in_images = []
        inputs = BatchFeature()

        if images is not None:
            if is_image_or_image_url(images):
                images = [[images]]
            elif isinstance(images, list) and is_image_or_image_url(images[0]):
                images = [images]
            elif (
                not isinstance(images, list)
                and not isinstance(images[0], list)
                and not is_image_or_image_url(images[0][0])
            ):
                raise ValueError(
                    "Invalid input images. Please provide a single image or a list of images or a list of list of images."
                )
            n_images_in_images = [len(sample) for sample in images]

            # Load images if they are URLs
            new_images = []
            for sample in images:
                new_images.append([])
                for im in sample:
                    if is_valid_image(im):
                        new_images[-1].append(im) # already loaded
                    elif isinstance(im, str):
                        new_images[-1].append(load_image(im))

            images = new_images
            del new_images

            image_inputs = self.image_processor(images, return_tensors=return_tensors, return_row_col_info=True)
            inputs.update(image_inputs)

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list) and not isinstance(text[0], str):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")

            image_rows = inputs.pop("rows", [[0] * len(text)])
            image_cols = inputs.pop("cols", [[0] * len(text)])

            fake_image_token = self.fake_image_token.content
            image_token = self.image_token.content

            prompt_strings = []
            for sample, sample_rows, sample_cols in zip(text, image_rows, image_cols):
                n_images_in_text.append(sample.count(image_token))

                # Replace the image token with fake tokens around the expanded image token sequence of length `image_seq_len`
                image_prompt_strings = []
                for n_rows, n_cols in zip(sample_rows, sample_cols):
                    image_prompt_string = get_image_prompt_string(
                        n_rows,
                        n_cols,
                        image_seq_len,
                        image_token=image_token,
                        fake_token_around_image=fake_image_token,
                    )
                    image_prompt_strings.append(image_prompt_string)

                split_sample = sample.split(image_token)

                # Place in the image prompt strings where the image tokens are
                sample = split_sample[0]
                for i, image_prompt_string in enumerate(image_prompt_strings):
                    sample += image_prompt_string + split_sample[i + 1]
                prompt_strings.append(sample)

            text_inputs = self.tokenizer(
                text=prompt_strings,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                is_split_into_words=is_split_into_words,
                return_tensors=return_tensors,
            )
            inputs.update(text_inputs)

            if n_images_in_images != n_images_in_text:
                raise ValueError(
                    f"The number of images in the text {n_images_in_text} and images  {n_images_in_images} should be the same."
                )

        return inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Idefics3TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Idefics3TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
