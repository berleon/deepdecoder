import os
import click
from deepdecoder.data import HDF5Dataset
import progressbar


@click.command()
@click.option('-o', '--output', type=click.Path())
@click.option('--batch-size', default=256, type=int)
@click.argument('hdf5', type=click.Path(dir_okay=False))
def main(output, batch_size, hdf5):
    if os.path.exists(output):
        os.remove(output)
    h5 = HDF5Dataset(hdf5)
    h5._dataset_created = True
    print("Shuffling: {}".format(hdf5))
    name = h5.dataset_names[0]
    nb_samples = len(h5[name])
    shuffled_h5 = HDF5Dataset(output, nb_samples=nb_samples)
    for key in h5.attrs.keys():
        if key.startswith('_'):
            continue
        shuffled_h5.attrs[key] = h5.attrs[key]

    bar = progressbar.ProgressBar(max_value=int(nb_samples))
    nb_seen = 0
    for i, batch in enumerate(h5.iter(batch_size, shuffle=True)):
        nb = min(batch_size, nb_samples - nb_seen)
        append_batch = {}
        for name, arr in batch.items():
            append_batch[name] = arr[:nb]
        shuffled_h5.append(**batch)
        nb_seen += nb
        bar.update(nb_seen)
        if nb_seen >= nb_samples:
            return
    print("Shuffled dataset saved to: {}".format(output))
