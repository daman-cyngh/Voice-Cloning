from synthesizer.hparams import hparams as _syn_hp


# Audio settings------------------------------------------------------------------------
# Match the values of the synthesizer
sample_rate = _syn_hp.sample_rate
n_fft = _syn_hp.n_fft
num_mels = _syn_hp.num_mels
hop_length = _syn_hp.hop_size
win_length = _syn_hp.win_size
fmin = _syn_hp.fmin
min_level_db = _syn_hp.min_level_db
ref_level_db = _syn_hp.ref_level_db
mel_max_abs_value = _syn_hp.max_abs_value
preemphasis = _syn_hp.preemphasis
apply_preemphasis = _syn_hp.preemphasize

bits = 9                            
mu_law = True                       


voc_mode = 'RAW'                  
voc_upsample_factors = (5, 5, 8)   
voc_rnn_dims = 512
voc_fc_dims = 512
voc_compute_dims = 128
voc_res_out_dims = 128
voc_res_blocks = 10


voc_batch_size = 100
voc_lr = 1e-4
voc_gen_at_checkpoint = 5           
voc_pad = 2                         
voc_seq_len = hop_length * 5        


voc_gen_batched = True              
voc_target = 8000                   
voc_overlap = 400                   
