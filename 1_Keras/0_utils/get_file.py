def get_file(fname='imdb', path='/fbidm/CLOUD_DISK/notebook/PyTest'):
    from pathlib import Path
    import keras.datasets as kd
    p = Path(path)
    assert p.is_dir() == True
    
    _kd = kd.__getattribute__(fname)
    _kd.get_file('%s.npz' % fname, list(p.glob('%s.*' % fname))[0].as_uri())
    return _kd.load_data()
