from absl import flags


def define_flags():
    flags.DEFINE_string('run', None, 'Experiment run')
    flags.mark_flag_as_required('run')

    # Hyperparams
    flags.DEFINE_integer('batch_size', 2, 'Batch size')
    flags.DEFINE_integer('epoch', 10, 'Epoch number')
    flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
    flags.DEFINE_boolean('cont', False, 'Continue training from ckpt')

    # Debug
    flags.DEFINE_integer('debug_freq', -1, 'Debug output freq')

    # Data
    flags.DEFINE_string('input', '/Users/metuoku/data/cityscapes/', 'Cityscapes input folder')
    flags.DEFINE_list('resolution', ['128', '256'], 'Resolution')
    flags.DEFINE_integer('limit', -1, 'Limit')

    # Augmentation
    flags.DEFINE_float('flip_prob', 0.0, 'Augmentation flip probability')
    flags.DEFINE_float('crop', 0.0, 'Augmentation crop ratio')
    flags.DEFINE_float('crop_offset', 0.0, 'Augmentation crop offset')

    # MixMatch
    flags.DEFINE_integer('unlabeled_batch_size', 2, 'Unlabeled batch size')
    flags.DEFINE_integer('K', 2, 'Augmentation ratio')
    flags.DEFINE_float('loss_modifier', 100, 'Semisupervised loss modifier')
    flags.DEFINE_float('temperature', 0.5, 'Sharpening temperature')
    flags.DEFINE_float('alpha', 0.75, 'MixUp blending distribution parameter')
    flags.DEFINE_boolean('do_mixup', False, 'MixUp labels or not')
