from tensorflow.contrib.training import HParams


hparams = HParams(
    
    cleaners="english_cleaners",
    
    tacotron_gpu_start_idx=0,  
    tacotron_num_gpus=1,  
    split_on_cpu=True,
    
    num_mels=80,  
    
    rescale=True,  
    rescaling_max=0.9,  
    
    clip_mels_length=True,
    
    max_mel_frames=900,
   
    use_lws=False,
    
    silence_threshold=2,  
    
    n_fft=800,  
    hop_size=200,  
    win_size=800,  
    sample_rate=16000,  
    
    frame_shift_ms=None,  
    
    trim_fft_size=512,
    trim_hop_size=128,
    trim_top_db=23,
    
    signal_normalization=True,
    
    allow_clipping_in_normalization=True,  
    symmetric_mels=True,
    
    max_abs_value=4.,
    
    normalize_for_wavenet=True,
    
    clip_for_wavenet=True,
    
    preemphasize=True,  
    preemphasis=0.97,  
    
    min_level_db=-100,
    ref_level_db=20,
    fmin=55,
    
    fmax=7600,  
    
    power=1.5,
    
    griffin_lim_iters=60,
    
    outputs_per_step=2, 
    
    stop_at_any=True,
    
    embedding_dim=512,  
    
    
    enc_conv_num_layers=3,  
    enc_conv_kernel_size=(5,),  
    enc_conv_channels=512,  
    encoder_lstm_units=256,  
    
    
    smoothing=False,  
    attention_dim=128,  
    attention_filters=32,  
    attention_kernel=(31,),  
    cumulative_weights=True,
    
    prenet_layers=[256, 256],  
    decoder_layers=2,  
    decoder_lstm_units=1024,  
    max_iters=2000,
    
    postnet_num_layers=5,  
    postnet_kernel_size=(5,),  
    postnet_channels=512,  
    
    cbhg_kernels=8,
    
    cbhg_conv_channels=128,  
    cbhg_pool_size=2,  
    cbhg_projection=256,
    
    cbhg_projection_kernel_size=3,  
    cbhg_highwaynet_layers=4,  
    cbhg_highway_units=128,  
    cbhg_rnn_units=128,
    
    mask_encoder=True,
    
    mask_decoder=False,
    
    cross_entropy_pos_weight=20,
    
    predict_linear=False,
    
    tacotron_random_seed=5339,
    
    tacotron_data_random_state=1234,  
    
    
    tacotron_swap_with_cpu=False,
    
    tacotron_batch_size=36,  
    
    tacotron_synthesis_batch_size=128,
    
    tacotron_test_size=0.05,
    
    tacotron_test_batches=None,  
    
    
    tacotron_decay_learning_rate=True,
    tacotron_start_decay=50000,  
    tacotron_decay_steps=50000,  
    tacotron_decay_rate=0.5,  
    tacotron_initial_learning_rate=1e-3,  
    tacotron_final_learning_rate=1e-5,  
    
    
    tacotron_adam_beta1=0.9,  
    tacotron_adam_beta2=0.999,  
    tacotron_adam_epsilon=1e-6,  
    
    
    tacotron_reg_weight=1e-7,  
    tacotron_scale_regularization=False,
    
    tacotron_zoneout_rate=0.1, 
    tacotron_dropout_rate=0.5,  
    tacotron_clip_gradients=True,  
    
    
    natural_eval=False,
    
    tacotron_teacher_forcing_mode="constant",
    
    tacotron_teacher_forcing_ratio=1.,
    
    tacotron_teacher_forcing_init_ratio=1.,
    
    tacotron_teacher_forcing_final_ratio=0.,
    
    tacotron_teacher_forcing_start_decay=10000,
    
    tacotron_teacher_forcing_decay_steps=280000,
    
    tacotron_teacher_forcing_decay_alpha=0.,
    
    train_with_GTA=False,
    
    sentences=[
        
        "Scientists at the CERN laboratory say they have discovered a new particle.",
        "There\"s a way to measure the acute emotional intelligence that has never gone out of "
		"style.",
        "President Trump met with other leaders at the Group of 20 conference.",
        "The Senate\"s bill to repeal and replace the Affordable Care Act is now imperiled.",
        
        "Generative adversarial network or variational auto-encoder.",
        "Basilar membrane and otolaryngology are not auto-correlations.",
        "He has read the whole thing.",
        "He reads books.",
        "He thought it was time to present the present.",
        "Thisss isrealy awhsome.",
        "Punctuation sensitivity, is working.",
        "Punctuation sensitivity is working.",
        "Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?",
        "She sells sea-shells on the sea-shore. The shells she sells are sea-shells I'm sure.",
        "Tajima Airport serves Toyooka.",
        
        "Sequence to sequence models have enjoyed great success in a variety of tasks such as machine translation, speech recognition, and text summarization.\
        This project covers a sequence to sequence model trained to predict a speech representation from an input sequence of characters. We show that\
        the adopted architecture is able to perform this task with wild success.",
        "Thank you so much for your support!",
    ],
    

    speaker_embedding_size=256,
    silence_min_duration_split=0.4, 
    utterance_min_duration=1.6,     
    
)


def hparams_debug_string():
    values = hparams.values()
    hp = ["  %s: %s" % (name, values[name]) for name in sorted(values) if name != "sentences"]
    return "Hyperparameters:\n" + "\n".join(hp)
