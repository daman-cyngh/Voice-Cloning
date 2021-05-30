from synthesizer.tacotron2 import Tacotron2
from synthesizer.hparams import hparams
from multiprocess.pool import Pool  
from synthesizer import audio
from pathlib import Path
from typing import Union, List
import tensorflow as tf
import numpy as np
import numba.cuda
import librosa


class Synthesizer:
    sample_rate = hparams.sample_rate
    hparams = hparams
    
    def __init__(self, checkpoints_dir: Path, verbose=True, low_mem=False):
        
        self.verbose = verbose
        self._low_mem = low_mem
        
        
        self._model = None  
        checkpoint_state = tf.train.get_checkpoint_state(checkpoints_dir)
        if checkpoint_state is None:
            raise Exception("Could not find any synthesizer weights under %s" % checkpoints_dir)
        self.checkpoint_fpath = checkpoint_state.model_checkpoint_path
        if verbose:
            model_name = checkpoints_dir.parent.name.replace("logs-", "")
            step = int(self.checkpoint_fpath[self.checkpoint_fpath.rfind('-') + 1:])
            print("Found synthesizer \"%s\" trained to step %d" % (model_name, step))
     
    def is_loaded(self):
        
        return self._model is not None
    
    def load(self):
        
        if self._low_mem:
            raise Exception("Cannot load the synthesizer permanently in low mem mode")
        tf.reset_default_graph()
        self._model = Tacotron2(self.checkpoint_fpath, hparams)
            
    def synthesize_spectrograms(self, texts: List[str],
                                embeddings: Union[np.ndarray, List[np.ndarray]],
                                return_alignments=False):
        
        if not self._low_mem:
            if not self.is_loaded():
                self.load()
            specs, alignments = self._model.my_synthesize(embeddings, texts)
        else:
            
            specs, alignments = Pool(1).starmap(Synthesizer._one_shot_synthesize_spectrograms, 
                                                [(self.checkpoint_fpath, embeddings, texts)])[0]
    
        return (specs, alignments) if return_alignments else specs

    @staticmethod
    def _one_shot_synthesize_spectrograms(checkpoint_fpath, embeddings, texts):
        
        tf.reset_default_graph()
        model = Tacotron2(checkpoint_fpath, hparams)
        specs, alignments = model.my_synthesize(embeddings, texts)
        
        
        specs, alignments = [spec.copy() for spec in specs], alignments.copy()
        
        
        model.session.close()
        numba.cuda.select_device(0)
        numba.cuda.close()
        
        return specs, alignments

    @staticmethod
    def load_preprocess_wav(fpath):
        
        wav = librosa.load(fpath, hparams.sample_rate)[0]
        if hparams.rescale:
            wav = wav / np.abs(wav).max() * hparams.rescaling_max
        return wav

    @staticmethod
    def make_spectrogram(fpath_or_wav: Union[str, Path, np.ndarray]):
        
        if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
            wav = Synthesizer.load_preprocess_wav(fpath_or_wav)
        else:
            wav = fpath_or_wav
        
        mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
        return mel_spectrogram
    
    @staticmethod
    def griffin_lim(mel):
        
        return audio.inv_mel_spectrogram(mel, hparams)
    