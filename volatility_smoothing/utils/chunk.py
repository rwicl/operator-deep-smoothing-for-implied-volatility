from typing import Sequence


def chunked(sequence: Sequence, chunk_size: int) -> list:
    """Chunk a sequence into chunks of size `chunk_size`.

    Parameters
    ----------
    sequence
        Sequence to chunk
    chunk_size
        Size of chunks

    Returns
    -------
    List
        List of chunks
    """
    return [sequence[pos:pos + chunk_size] for pos in range(0, len(sequence), chunk_size)]