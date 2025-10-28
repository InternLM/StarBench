from abc import abstractmethod
import librosa
import torch
from loguru import logger


class BaseModel:

    NAME = None

    @abstractmethod
    def generate_inner(self, msgs: dict) -> (str, str):
        raise NotImplementedError


    @torch.inference_mode()
    def __call__(self, msgs: dict) -> str:
        return self.generate_inner(msgs)