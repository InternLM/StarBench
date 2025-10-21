from abc import abstractmethod
import librosa
import torch
from loguru import logger


class BaseModel:

    NAME = None


    def use_custom_prompt(self, dataset):
        """Whether to use custom prompt for the given dataset.

        Args:
            dataset (str): The name of the dataset.

        Returns:
            bool: Whether to use custom prompt. If True, will call `build_prompt` of the VLM to build the prompt.
                Default to False.
        """
        return False

    @abstractmethod
    def build_prompt(self, line, dataset):
        """Build custom prompts for a specific dataset. Called only if `use_custom_prompt` returns True.

        Args:
            line (line of pd.DataFrame): The raw input line.
            dataset (str): The name of the dataset.

        Returns:
            str: The built message.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_inner(self, msgs: dict) -> (str, str):
        raise NotImplementedError

    # @staticmethod
    # def check_audio_legal(audio_path: str | list[str], max_duration: float = 60) -> bool:
    #     """by default, we discard audio longer than 60s. subclasses can override this method (depends on model requirements)
    #     """
    #     if isinstance(audio_path, str):
    #         duration = librosa.get_duration(path=audio_path)
    #         if duration > max_duration or duration < 0.1:
    #             return False
    #     else:
    #         for path in audio_path:
    #             duration = librosa.get_duration(path=path)
    #             if duration > max_duration or duration < 0.1:
    #                 return False
    #     return True

    @torch.inference_mode()
    def __call__(self, msgs: dict) -> str:
        # if not self.check_audio_legal(msg['audio']):
        #     logger.warning(
        #         f'dataset: {msg["meta"]["dataset_name"]}, audio: {msg["audio"]}, duration exceeds 60s limit, skipping this sample')
        #     return msg['text'], None
        return self.generate_inner(msgs)